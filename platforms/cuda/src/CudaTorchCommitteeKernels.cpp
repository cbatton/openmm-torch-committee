/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2018-2024 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors: Raimondas Galvelis, Raul P. Pelaez                           *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "CudaTorchCommitteeKernels.h"
#include "CudaTorchCommitteeKernelSources.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/internal/ContextImpl.h"
#include <map>
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
using namespace TorchCPlugin;
using namespace OpenMM;
using namespace std;

// macro for checking the result of synchronization operation on CUDA
// copied from `openmm/platforms/cuda/src/CudaParallelKernels.cpp`
#define CHECK_RESULT(result, prefix)                                             \
    if (result != CUDA_SUCCESS) {                                                \
        std::stringstream m;                                                     \
        m << prefix << ": " << cu.getErrorString(result) << " (" << result << ")"\
          << " at " << __FILE__ << ":" << __LINE__;                              \
        throw OpenMMException(m.str());                                          \
    }

CudaCalcTorchForceCommitteeKernel::CudaCalcTorchForceCommitteeKernel(string name, const Platform& platform, CudaContext& cu) : CalcTorchForceCommitteeKernel(name, platform), hasInitializedKernel(false), cu(cu) {
    // Explicitly activate the primary context
    CHECK_RESULT(cuDevicePrimaryCtxRetain(&primaryContext, cu.getDevice()), "Failed to retain the primary context");
}

CudaCalcTorchForceCommitteeKernel::~CudaCalcTorchForceCommitteeKernel() {
    cuDevicePrimaryCtxRelease(cu.getDevice());
}

void CudaCalcTorchForceCommitteeKernel::initialize(const System& system, const TorchForceCommittee& force, torch::jit::script::Module& module, const c10::intrusive_ptr<c10d::Backend>& mpi_group) {
    this->module = module;
    m_mpi_group = mpi_group;
    usePeriodic = force.usesPeriodicBoundaryConditions();
    outputsForces = force.getOutputsForces();
    for (int i = 0; i < force.getNumGlobalParameters(); i++)
        globalNames.push_back(force.getGlobalParameterName(i));
    for (int i = 0; i < force.getNumEnergyParameterDerivatives(); i++) {
        paramDerivs.insert(force.getEnergyParameterDerivativeName(i));
        cu.addEnergyParameterDerivative(force.getEnergyParameterDerivativeName(i));
    }
    int numParticles = system.getNumParticles();

    // Push the PyTorch context
    // NOTE: Pytorch is always using the primary context.
    //       It makes the primary context current, if it is not a case.
    CHECK_RESULT(cuCtxPushCurrent(primaryContext), "Failed to push the CUDA context");

    // Initialize CUDA objects for PyTorch
    const torch::Device device(torch::kCUDA, cu.getDeviceIndex()); // This implicitly initialize PyTorch
    this->module.to(device);
    this->module.eval();
    this->module = torch::jit::freeze(this->module);
    torch::TensorOptions options = torch::TensorOptions().device(device).dtype(cu.getUseDoublePrecision() ? torch::kFloat64 : torch::kFloat32);
    posTensor = torch::empty({numParticles, 3}, options.requires_grad(!outputsForces));
    boxTensor = torch::empty({3, 3}, options);
    energyTensor = torch::empty({0}, options);
    forceTensor = torch::empty({0}, options);
    for (const string& name : globalNames)
        globalTensors[name] = torch::tensor({0}, options);
    // Pop the PyToch context
    CUcontext ctx;
    CHECK_RESULT(cuCtxPopCurrent(&ctx), "Failed to pop the CUDA context");
    assert(primaryContext == ctx); // Check that PyTorch haven't messed up the context stack
    
    // Get process group information
    rank = m_mpi_group->getRank();
    world_size = m_mpi_group->getSize();

    // Initialize CUDA objects for OpenMM-Torch
    ContextSelector selector(cu); // Switch to the OpenMM context
    map<string, string> defines;
    CUmodule program = cu.createModule(CudaTorchCommitteeKernelSources::torchForce, defines);
    copyInputsKernel = cu.getKernel(program, "copyInputs");
    addForcesKernel = cu.getKernel(program, "addForces");
    auto properties = force.getProperties();
    const std::string useCUDAGraphsString = properties["useCUDAGraphs"];
    if (useCUDAGraphsString == "true")
        useGraphs = true;
    else if (useCUDAGraphsString == "false" || useCUDAGraphsString == "")
        useGraphs = false;
    else
        throw OpenMMException("TorchForce: invalid value of \"useCUDAGraphs\"");
    if (useGraphs) {
        this->warmupSteps = std::stoi(properties["CUDAGraphWarmupSteps"]);
        if (this->warmupSteps <= 0) {
            throw OpenMMException("TorchForce: \"CUDAGraphWarmupSteps\" must be a positive integer");
        }
    }
}

/**
 * Get a pointer to the data in a PyTorch tensor.
 * The tensor is converted to the correct data type if necessary.
 */
static void* getTensorPointer(OpenMM::CudaContext& cu, torch::Tensor& tensor) {
    void* data;
    if (cu.getUseDoublePrecision()) {
        data = tensor.to(torch::kFloat64).data_ptr<double>();
    } else {
        data = tensor.to(torch::kFloat32).data_ptr<float>();
    }
    return data;
}

/**
 * Prepare the inputs for the PyTorch model, copying positions from the OpenMM context.
 */
void CudaCalcTorchForceCommitteeKernel::prepareTorchInputs(ContextImpl& context, vector<torch::jit::IValue>& inputs, map<string, torch::Tensor>& globalTensors) {
    int numParticles = cu.getNumAtoms();
    // Get pointers to the atomic positions and simulation box
    void* posData = getTensorPointer(cu, posTensor);
    void* boxData = getTensorPointer(cu, boxTensor);
    // Copy the atomic positions and simulation box to PyTorch tensors
    {
        ContextSelector selector(cu); // Switch to the OpenMM context
        void* inputArgs[] = {&posData,
                             &boxData,
                             &cu.getPosq().getDevicePointer(),
                             &cu.getAtomIndexArray().getDevicePointer(),
                             &numParticles,
                             cu.getPeriodicBoxVecXPointer(),
                             cu.getPeriodicBoxVecYPointer(),
                             cu.getPeriodicBoxVecZPointer()};
        cu.executeKernel(copyInputsKernel, inputArgs, numParticles);
        CHECK_RESULT(cuCtxSynchronize(), "Failed to synchronize the CUDA context"); // Synchronize before switching to the PyTorch context
    }
    // Communicate posTensor, boxTensor between MPI ranks
    // Broadcast from rank 0 to all other ranks
    vector<torch::Tensor> posTensors = {posTensor};
    auto work = m_mpi_group->broadcast(posTensors, {.rootRank = 0});
    work->wait();
    if (usePeriodic) {
        vector<torch::Tensor> boxTensors = {boxTensor};
        auto work = m_mpi_group->broadcast(boxTensors, {.rootRank = 0});
        work->wait();
    }
    // Prepare the input of the PyTorch model
    inputs = {posTensor};
    if (usePeriodic)
        inputs.push_back(boxTensor);
    for (const string& name : globalNames) {
        // PyTorch requires us to set requires_grad to false before initializing a tensor.
        globalTensors[name].set_requires_grad(false);
        globalTensors[name][0] = context.getParameter(name);
        if (paramDerivs.find(name) != paramDerivs.end())
            globalTensors[name].set_requires_grad(true);
        inputs.push_back(globalTensors[name]);
    }
}

/**
 * Add the computed forces to the total atomic forces.
 */
void CudaCalcTorchForceCommitteeKernel::addForces(torch::Tensor& forceTensor) {
    int numParticles = cu.getNumAtoms();
    // Get a pointer to the computed forces
    void* forceData = getTensorPointer(cu, forceTensor);
    CHECK_RESULT(cuCtxSynchronize(), "Failed to synchronize the CUDA context"); // Synchronize before switching to the OpenMM context
    // Add the computed forces to the total atomic forces
    {
        ContextSelector selector(cu); // Switch to the OpenMM context
        int paddedNumAtoms = cu.getPaddedNumAtoms();
        int forceSign = (outputsForces ? 1 : -1);
        void* forceArgs[] = {&forceData, &cu.getForce().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(), &numParticles, &paddedNumAtoms, &forceSign};
        cu.executeKernel(addForcesKernel, forceArgs, numParticles);
        CHECK_RESULT(cuCtxSynchronize(), "Failed to synchronize the CUDA context"); // Synchronize before switching to the PyTorch context
    }
}

/**
 * This function launches the workload in a way compatible with CUDA
 * graphs as far as OpenMM-Torch goes.  Capturing this function when
 * the model is not itself graph compatible (due to, for instance,
 * implicit synchronizations) will result in a CUDA error.
 */
static void executeGraph(bool outputsForces, bool includeForces, torch::jit::script::Module& module, vector<torch::jit::IValue>& inputs, torch::Tensor& posTensor, torch::Tensor& energyTensor,
                         torch::Tensor& forceTensor, map<string, torch::Tensor>& globalTensors, set<string> paramDerivs) {
    vector<torch::Tensor> gradInputs;
    if (!outputsForces && includeForces)
        gradInputs.push_back(posTensor);
    for (auto& name : paramDerivs)
        gradInputs.push_back(globalTensors[name]);
    auto none = torch::Tensor();
    if (outputsForces) {
        auto outputs = module.forward(inputs).toTuple();
        energyTensor = outputs->elements()[0].toTensor();
        forceTensor = outputs->elements()[1].toTensor();
        if (gradInputs.size() > 0)
            energyTensor.backward(none, false, false, gradInputs);
    } else {
        energyTensor = module.forward(inputs).toTensor();
        // Compute force by backpropagating the PyTorch model
        // CUDA graph capture sometimes fails if backwards is not explicitly requested w.r.t positions
        // See https://github.com/openmm/openmm-torch/pull/120/
        if (gradInputs.size() > 0)
            energyTensor.backward(none, false, false, gradInputs);
        if (includeForces) {
            // This is minus the forces, we change the sign later on
            forceTensor = posTensor.grad().clone();
            // Zero the gradient to avoid accumulating it
            posTensor.grad().zero_();
        }
    }
}

double CudaCalcTorchForceCommitteeKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    // Push to the PyTorch context
    CHECK_RESULT(cuCtxPushCurrent(primaryContext), "Failed to push the CUDA context");
    vector<torch::jit::IValue> inputs;
    prepareTorchInputs(context, inputs, globalTensors);
    if (!useGraphs) {
        executeGraph(outputsForces, includeForces, module, inputs, posTensor, energyTensor, forceTensor, globalTensors, paramDerivs);
    } else {
        // Record graph if not already done
        bool is_graph_captured = false;
        if (graphs.find(includeForces) == graphs.end()) {
	    //CUDA graph capture must occur in a non-default stream
            const auto stream = c10::cuda::getStreamFromPool(false, cu.getDeviceIndex());
	        const c10::cuda::CUDAStreamGuard guard(stream);
            // Warmup the graph workload before capturing.  This first
            // run  before  capture sets  up  allocations  so that  no
            // allocations are  needed after.  Pytorch's  allocator is
            // stream  capture-aware and,  after warmup,  will provide
            // record static pointers and shapes during capture.
            try {
                for (int i = 0; i < this->warmupSteps; i++)
                    executeGraph(outputsForces, includeForces, module, inputs, posTensor, energyTensor, forceTensor, globalTensors, paramDerivs);
            }
            catch (std::exception& e) {
                throw OpenMMException(string("TorchForce Failed to warmup the model before graph construction. Torch reported the following error:\n") + e.what());
            }
            graphs[includeForces].capture_begin();
            try {
                executeGraph(outputsForces, includeForces, module, inputs, posTensor, energyTensor, forceTensor, globalTensors, paramDerivs);
                is_graph_captured = true;
                graphs[includeForces].capture_end();
            }
            catch (std::exception& e) {
                if (!is_graph_captured) {
                    graphs[includeForces].capture_end();
                }
                throw OpenMMException(string("TorchForce Failed to capture the model into a CUDA graph. Torch reported the following error:\n") + e.what());
            }
            for (const string& name : paramDerivs)
                globalTensors[name].grad().zero_();
        }
        // Use the same stream as the OpenMM context, even if it is the default stream
        const auto openmmStream = cu.getCurrentStream();
        const auto stream = c10::cuda::getStreamFromExternal(openmmStream, cu.getDeviceIndex());
        const c10::cuda::CUDAStreamGuard guard(stream);
        graphs[includeForces].replay();
    }
    c10d::AllreduceOptions opts_c10d;
    opts_c10d.reduceOp = c10d::ReduceOp::AVG;
    if (includeForces) {
        // do all_reduce on forces and average
        vector<torch::Tensor> forceTensors = {forceTensor};
        auto work = m_mpi_group->allreduce(forceTensors, opts_c10d);
        work->wait();
        addForces(forceTensor);
    }
    map<string, double>& energyParamDerivs = cu.getEnergyParamDerivWorkspace();
    for (const string& name : paramDerivs) {
        energyParamDerivs[name] += globalTensors[name].grad().item<double>();
        globalTensors[name].grad().zero_();
    }
    // Get energy
    vector<torch::Tensor> energyTensors = {energyTensor};
    auto work = m_mpi_group->allreduce(energyTensors, opts_c10d);
    work->wait();
    const double energy = energyTensor.item<double>(); // This implicitly synchronizes the PyTorch context
    // Pop to the PyTorch context
    CUcontext ctx;
    CHECK_RESULT(cuCtxPopCurrent(&ctx), "Failed to pop the CUDA context");
    assert(primaryContext == ctx); // Check that the correct context was popped
    return energy;
}
