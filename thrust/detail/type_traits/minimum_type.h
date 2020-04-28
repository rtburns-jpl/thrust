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

#include <thrust/detail/type_traits.h>

namespace thrust
{

namespace detail
{ 

namespace minimum_type_detail
{

//
// Returns the minimum type or is empty
// if T1 and T2 are unrelated.
//
template <typename T1, typename T2, bool GreaterEqual, bool LessEqual> struct minimum_type_impl {};
  
template <typename T1, typename T2>
struct minimum_type_impl<T1,T2,true,false>
{
  typedef T2 type;
}; // end minimum_type_impl

template <typename T1, typename T2>
struct minimum_type_impl<T1,T2,false,true>
{
  typedef T1 type;
}; // end minimum_type_impl

template <typename T1, typename T2>
struct minimum_type_impl<T1,T2,true,true>
{
  typedef T1 type;
}; // end minimum_type_impl

template <typename T1, typename T2>
struct primitive_minimum_type
  : minimum_type_detail::minimum_type_impl<
      T1,
      T2,
      ::thrust::detail::is_convertible<T1,T2>::value,
      ::thrust::detail::is_convertible<T2,T1>::value
    >
{
}; // end primitive_minimum_type

// because some types are not convertible (even to themselves)
// specialize primitive_minimum_type for when both types are identical
template <typename T>
struct primitive_minimum_type<T,T>
{
  typedef T type;
}; // end primitive_minimum_type

// XXX this belongs somewhere more general
struct any_conversion
{
  template<typename T> operator T (void);
};

} // end minimum_type_detail

template<typename T, typename... Ts>
  struct minimum_type;

template<typename T>
  struct minimum_type<T>
{
  typedef T type;
};

// base case
template<typename T1, typename T2>
  struct minimum_type<T1,T2>
    : minimum_type_detail::primitive_minimum_type<T1,T2>
{};

template<typename T1, typename T2>
  struct lazy_minimum_type
    : minimum_type<
        typename T1::type,
        typename T2::type
      >
{};

// carefully avoid referring to a nested ::type which may not exist
template<typename T1, typename T2, typename T3, typename... Ts>
  struct minimum_type<T1, T2, T3, Ts...>
    : lazy_minimum_type<
        minimum_type<
          T1, T2
        >,
        minimum_type<
          T3, Ts...
        >
      >
{};

} // end detail

} // end thrust

