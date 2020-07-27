# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2020 Max-Planck-Society


import ducc0.misc
import numpy as np
import pytest
import time
from numpy.testing import assert_

transpose = ducc0.misc.transpose

pmp = pytest.mark.parametrize

def transpose_classic(arr, axes):
    return np.ascontiguousarray(np.transpose(arr, axes))

def transpose_new(arr, axes):
    axinv = np.argsort(axes)
    b = np.empty(np.array(arr.shape)[axes],dtype=arr.dtype)
    b = np.transpose(b, axinv)
    return np.transpose(transpose(arr,b),axes)


def test1():
    rng = np.random.default_rng(42)
    for i in range(1000):
        ndim = rng.integers(2, 3)
        axlen = max(500,int((2**20)**(1./ndim)))
        shape = rng.integers(2, axlen, ndim)
        axes = np.arange(ndim)
        rng.shuffle(axes)
        if np.all(axes == np.arange(ndim)):
            continue
        a = rng.random(shape)-0.5
        t0 = time.time()
        c = transpose_classic(a, axes)
        ta = time.time()-t0
        t0 = time.time()
        b = transpose_new(a,axes)
        tb = time.time()-t0
        assert(c.shape==b.shape)
        assert(c.strides==b.strides)
        assert(np.all(b==c))

if __name__ == "__main__":
    test1()
