/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/is_metafunction_defined.h>
#include <thrust/detail/type_traits/minimum_type.h>

namespace thrust
{
namespace detail
{

template<typename... Ts>
  struct unrelated_systems {};

// if a minimum system exists for these arguments, return it
// otherwise, collect the arguments and report them as unrelated
template<typename... Ts>
  struct minimum_system
    : thrust::detail::eval_if<
        is_metafunction_defined<
          minimum_type<Ts...>
        >::value,
        minimum_type<Ts...>,
        thrust::detail::identity_<
          unrelated_systems<Ts...>
        >
      >
{}; // end minimum_system

} // end detail
} // end thrust
