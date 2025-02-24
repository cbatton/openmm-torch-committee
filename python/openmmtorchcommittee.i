%module openmmtorchcommittee

%include "factory.i"
%import(module="simtk.openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include <std_string.i>
%include <std_map.i>

%{
#include "TorchForceCommittee.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
#include <torch/csrc/jit/python/module_python.h>
#include <torch/csrc/jit/serialization/import.h>
#include <c10d/ProcessGroupNCCL.hpp>
#include <c10d/ProcessGroupMPI.hpp>
#include <c10d/Backend.hpp>
#include <c10/util/intrusive_ptr.h>
#include <memory>
%}

// Tell SWIG about the c10d::Backend class
namespace c10d {
    class Backend;
}

// Ignore c10d::Backend, intrusive pointers
%ignore TorchCPlugin::TorchForceCommittee::initializeBackend;
%ignore TorchCPlugin::TorchForceCommittee::getMPIGroup;

/*
 * Convert C++ exceptions to Python exceptions.
*/
%exception {
    try {
        $action
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_Exception, const_cast<char*>(e.what()));
        return NULL;
    }
}

%typemap(in) const torch::jit::Module&(torch::jit::Module mod) {
    py::object o = py::reinterpret_borrow<py::object>($input);
    py::object pybuffer = py::module::import("io").attr("BytesIO")();
    py::module::import("torch.jit").attr("save")(o, pybuffer);
    std::string s = py::cast<std::string>(pybuffer.attr("getvalue")());
    std::stringstream buffer(s);
    mod = torch::jit::load(buffer);
    $1 = &mod;
}

%typemap(out) const torch::jit::Module& {
    std::stringstream buffer;
    $1->save(buffer);
    auto pybuffer = py::module::import("io").attr("BytesIO")(py::bytes(buffer.str()));
    $result = py::module::import("torch.jit").attr("load")(pybuffer).release().ptr();
}

%typecheck(SWIG_TYPECHECK_POINTER) const torch::jit::Module& {
    py::object o = py::reinterpret_borrow<py::object>($input);
    py::handle ScriptModule = py::module::import("torch.jit").attr("ScriptModule");
    $1 = py::isinstance(o, ScriptModule);
}

namespace std {
    %template(property_map) map<string, string>;
}

namespace TorchCPlugin {

class TorchForceCommittee : public OpenMM::Force {
public:
    TorchForceCommittee(const std::string& file, const std::string& backend, const int rank, const int world_size, const std::string& master_addr, const int master_port, const std::map<std::string, std::string>& properties = {});
    TorchForceCommittee(const torch::jit::Module& module, const std::string& backend, const int rank, const int world_size, const std::string& master_addr, const int master_port, const std::map<std::string, std::string>& properties = {});
    const std::string& getFile() const;
    const torch::jit::Module& getModule() const;
    int getRank() const;
    int getWorldSize() const;
    void setUsesPeriodicBoundaryConditions(bool periodic);
    bool usesPeriodicBoundaryConditions() const;
    void setOutputsForces(bool);
    bool getOutputsForces() const;
    int getNumGlobalParameters() const;
    int getNumEnergyParameterDerivatives() const;
    int addGlobalParameter(const std::string& name, double defaultValue);
    const std::string& getGlobalParameterName(int index) const;
    void setGlobalParameterName(int index, const std::string& name);
    double getGlobalParameterDefaultValue(int index) const;
    void setGlobalParameterDefaultValue(int index, double defaultValue);
    void addEnergyParameterDerivative(const std::string& name);
    const std::string& getEnergyParameterDerivativeName(int index) const;
    void setProperty(const std::string& name, const std::string& value);
    const std::map<std::string, std::string>& getProperties() const;

    /*
     * Add methods for casting a Force to a TorchForceCommittee.
    */
    %extend {
        static TorchCPlugin::TorchForceCommittee& cast(OpenMM::Force& force) {
            return dynamic_cast<TorchCPlugin::TorchForceCommittee&>(force);
        }

        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<TorchCPlugin::TorchForceCommittee*>(&force) != NULL);
        }
    }
};

}
