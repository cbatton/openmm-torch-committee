#ifndef CUDA_TORCHC_KERNELS_H_
#define CUDA_TORCHC_KERNELS_H_

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

#include "TorchCommitteeKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include <torch/version.h>
#include <ATen/cuda/CUDAGraph.h>
#include <set>

namespace TorchCPlugin {

/**
 * This kernel is invoked by TorchForceCommittee to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcTorchForceCommitteeKernel : public CalcTorchForceCommitteeKernel {
public:
    CudaCalcTorchForceCommitteeKernel(std::string name, const OpenMM::Platform& platform, OpenMM::CudaContext& cu);
    ~CudaCalcTorchForceCommitteeKernel();
    /**
     * Initialize the kernel.
     *
     * @param system         the System this kernel will be applied to
     * @param force          the TorchForceCommittee this kernel will be used for
     * @param module         the PyTorch module to use for computing forces and energy
     */
    void initialize(const OpenMM::System& system, const TorchForceCommittee& force, torch::jit::script::Module& module, const std::shared_ptr<c10d::ProcessGroup>& mpi_group);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);

private:
    bool hasInitializedKernel;
    OpenMM::CudaContext& cu;
    torch::jit::script::Module module;
    std::shared_ptr<c10d::ProcessGroup> m_mpi_group;
    torch::Tensor posTensor, boxTensor;
    torch::Tensor energyTensor, forceTensor;
    std::map<std::string, torch::Tensor> globalTensors;
    std::vector<std::string> globalNames;
    std::set<std::string> paramDerivs;
    bool usePeriodic, outputsForces;
    int rank = 0;
    int world_size = 1;
    CUfunction copyInputsKernel, addForcesKernel;
    CUcontext primaryContext;
    std::map<bool, at::cuda::CUDAGraph> graphs;
    void prepareTorchInputs(OpenMM::ContextImpl& context, std::vector<torch::jit::IValue>& inputs, std::map<std::string, torch::Tensor>& derivInputs);
    bool useGraphs;
    void addForces(torch::Tensor& forceTensor);
    int warmupSteps;
};

} // namespace TorchCPlugin

#endif /*CUDA_TORCHC_KERNELS_H_*/
