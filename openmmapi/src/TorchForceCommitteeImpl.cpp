/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2018 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
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

#include "internal/TorchForceCommitteeImpl.h"
#include "TorchCommitteeKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include <torch/script.h>

using namespace TorchCPlugin;
using namespace OpenMM;
using namespace std;

TorchForceCommitteeImpl::TorchForceCommitteeImpl(const TorchForceCommittee& owner) : owner(owner) {
}

TorchForceCommitteeImpl::~TorchForceCommitteeImpl() {
}

void TorchForceCommitteeImpl::initialize(ContextImpl& context) {
    auto module = owner.getModule().clone();
    auto m_mpi_group = owner.getMPIGroup();
    // Create the kernel.
    kernel = context.getPlatform().createKernel(CalcTorchForceCommitteeKernel::Name(), context);
    kernel.getAs<CalcTorchForceCommitteeKernel>().initialize(context.getSystem(), owner, module, m_mpi_group);
}

double TorchForceCommitteeImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcTorchForceCommitteeKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

map<string, double> TorchForceCommitteeImpl::getDefaultParameters() {
    map<string, double> parameters;
    for (int i = 0; i < owner.getNumGlobalParameters(); i++)
        parameters[owner.getGlobalParameterName(i)] = owner.getGlobalParameterDefaultValue(i);
    return parameters;
}

std::vector<std::string> TorchForceCommitteeImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcTorchForceCommitteeKernel::Name());
    return names;
}
