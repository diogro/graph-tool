// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2026 Tiago de Paula Peixoto <tiago@skewed.de>
//
// This program is free software; you can redistribute it and/or modify it under
// the terms of the GNU Lesser General Public License as published by the Free
// Software Foundation; either version 3 of the License, or (at your option) any
// later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
// details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#include "demangle.hh"
#include <cxxabi.h>
#include <cstdlib>
#include <format>

std::string name_demangle(const std::string& name)
{
    int status = 0;
    char *realname = abi::__cxa_demangle(name.c_str(), 0, 0, &status);
    if (status != 0)
        return std::format("{} (cannot demangle symbol, error: {})", name, status);
    std::string ret(realname);
    std::free(realname);
    return ret;
}
