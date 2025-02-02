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

#include "TorchForceCommittee.h"
#include "internal/TorchForceCommitteeImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include <fstream>
#include <torch/torch.h>
#include <torch/csrc/jit/serialization/import.h>
#include <c10d/ProcessGroupNCCL.hpp>
#include <c10d/ProcessGroupMPI.hpp>
#include <c10d/ProcessGroup.hpp>

using namespace TorchCPlugin;
using namespace OpenMM;
using namespace std;

TorchForceCommittee::TorchForceCommittee(const torch::jit::Module& module, const map<string, string>& properties) : file(), usePeriodic(false), outputsForces(false), module(module), m_mpi_group(nullptr) {
    const std::map<std::string, std::string> defaultProperties = {{"useCUDAGraphs", "false"}, {"CUDAGraphWarmupSteps", "10"}};
    this->properties = defaultProperties;
    for (auto& property : properties) {
        if (defaultProperties.find(property.first) == defaultProperties.end())
            throw OpenMMException("TorchForceCommittee: Unknown property '" + property.first + "'");
        this->properties[property.first] = property.second;
    }
    //auto options = c10d::ProcessGroupNCCL::Options::create();
    //auto m_mpi_group_nccl = std::make_shared<c10d::ProcessGroupNCCL>(0, 1, options);
    rank = 0; // m_mpi_group->getRank();
    world_size = 1; // m_mpi_group->getSize();
}

TorchForceCommittee::TorchForceCommittee(const std::string& file, const map<string, string>& properties) : TorchForceCommittee(torch::jit::load(file), properties) {
    this->file = file;
}

const string& TorchForceCommittee::getFile() const {
    return file;
}

const torch::jit::Module& TorchForceCommittee::getModule() const {
    return this->module;
}

const std::shared_ptr<c10d::ProcessGroup>& TorchForceCommittee::getMPIGroup() const {
    return m_mpi_group;
}

ForceImpl* TorchForceCommittee::createImpl() const {
    return new TorchForceCommitteeImpl(*this);
}

void TorchForceCommittee::setUsesPeriodicBoundaryConditions(bool periodic) {
    usePeriodic = periodic;
}

bool TorchForceCommittee::usesPeriodicBoundaryConditions() const {
    return usePeriodic;
}

void TorchForceCommittee::setOutputsForces(bool outputsForces) {
    this->outputsForces = outputsForces;
}

bool TorchForceCommittee::getOutputsForces() const {
    return outputsForces;
}

int TorchForceCommittee::addGlobalParameter(const string& name, double defaultValue) {
    globalParameters.push_back(GlobalParameterInfo(name, defaultValue));
    return globalParameters.size() - 1;
}

int TorchForceCommittee::getNumGlobalParameters() const {
    return globalParameters.size();
}

const string& TorchForceCommittee::getGlobalParameterName(int index) const {
    ASSERT_VALID_INDEX(index, globalParameters);
    return globalParameters[index].name;
}

void TorchForceCommittee::setGlobalParameterName(int index, const string& name) {
    ASSERT_VALID_INDEX(index, globalParameters);
    globalParameters[index].name = name;
}

double TorchForceCommittee::getGlobalParameterDefaultValue(int index) const {
    ASSERT_VALID_INDEX(index, globalParameters);
    return globalParameters[index].defaultValue;
}

void TorchForceCommittee::setGlobalParameterDefaultValue(int index, double defaultValue) {
    ASSERT_VALID_INDEX(index, globalParameters);
    globalParameters[index].defaultValue = defaultValue;
}

int TorchForceCommittee::getNumEnergyParameterDerivatives() const {
    return energyParameterDerivatives.size();
}

void TorchForceCommittee::addEnergyParameterDerivative(const string& name) {
    for (int i = 0; i < globalParameters.size(); i++)
        if (name == globalParameters[i].name) {
            energyParameterDerivatives.push_back(i);
            return;
        }
    throw OpenMMException(string("addEnergyParameterDerivative: Unknown global parameter '"+name+"'"));
}

const string& TorchForceCommittee::getEnergyParameterDerivativeName(int index) const {
    ASSERT_VALID_INDEX(index, energyParameterDerivatives);
    return globalParameters[energyParameterDerivatives[index]].name;
}

void TorchForceCommittee::setProperty(const std::string& name, const std::string& value) {
    if (properties.find(name) == properties.end())
        throw OpenMMException("TorchForceCommittee: Unknown property '" + name + "'");
    properties[name] = value;
}

const std::map<std::string, std::string>& TorchForceCommittee::getProperties() const {
    return properties;
}
