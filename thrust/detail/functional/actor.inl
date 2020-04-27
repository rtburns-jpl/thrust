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

// Portions of this code are derived from
//
// Manjunath Kudlur's Carbon library
//
// and
//
// Based on Boost.Phoenix v1.2
// Copyright (c) 2001-2002 Joel de Guzman

#include <thrust/detail/config.h>
#include <thrust/detail/functional/composite.h>
#include <thrust/detail/functional/operators/assignment_operator.h>
#include <thrust/functional.h>

namespace thrust
{

namespace detail
{
namespace functional
{

template<typename Eval>
  __host__ __device__
  constexpr actor<Eval>
    ::actor()
      : eval_type()
{}

template<typename Eval>
  __host__ __device__
  actor<Eval>
    ::actor(const Eval &base)
      : eval_type(base)
{}

template<typename Eval>
  __host__ __device__
  typename apply_actor<
    typename actor<Eval>::eval_type,
    typename thrust::null_type
  >::type
    actor<Eval>
      ::operator()(void) const
{
  return eval_type::eval(thrust::null_type());
} // end basic_environment::operator()

template<typename Eval>
  template<typename... Ts>
    __host__ __device__
    typename apply_actor<
      typename actor<Eval>::eval_type,
      typename thrust::tuple<Ts&...>
    >::type
      actor<Eval>
        ::operator()(Ts&... ts) const
{
  return eval_type::eval(thrust::tie(ts...));
} // end basic_environment::operator()

template<typename Eval>
  template<typename T>
    __host__ __device__
    typename assign_result<Eval,T>::type
      actor<Eval>
        ::operator=(const T& _1) const
{
  return do_assign(*this,_1);
} // end actor::operator=()

} // end functional
} // end detail
} // end thrust
