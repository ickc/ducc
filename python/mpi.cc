/*
 *  This code is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This code is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this code; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/* Copyright (C) 2019-2020 Max-Planck-Society
   Author: Martin Reinecke */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "ducc0/bindings/pybind_utils.h"
#include "ducc0/infra/communication.h"

namespace ducc0 {

namespace detail_pymodule_mpi {

using namespace std;

namespace py = pybind11;

auto None = py::none();

Communicator getComm(const py::object &comm_)
  {
  auto pyMPI = py::module_::import("mpi4py.MPI");
  auto func = pyMPI.attr("_addressof");
  auto res = reinterpret_cast<const MPI_Comm *>(func(comm_).cast<size_t>());
  return Communicator(*res);
  }

void Pytest_comm(const py::object &comm_)
  {
  auto comm = getComm(comm_);
  cout << comm.rank() << "/" << comm.num_ranks() << endl;
  }

void add_mpi(py::module &msup)
  {
  using namespace pybind11::literals;
  auto m = msup.def_submodule("mpi");

  m.def("test_comm", &Pytest_comm, "comm"_a);
  }

}

using detail_pymodule_mpi::add_mpi;

}
