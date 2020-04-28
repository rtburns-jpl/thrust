/*
 *  Copyright 2008-2018 NVIDIA Corporation
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


/*! \file tuple.h
 *  \brief A type encapsulating a heterogeneous collection of elements
 */

/*
 * Copyright (C) 1999, 2000 Jaakko Järvi (jaakko.jarvi@cs.utu.fi)
 * 
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying NOTICE file for the complete license)
 *
 * For more information, see http://www.boost.org
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/tuple.inl>
#include <thrust/pair.h>

namespace thrust
{

/*! \addtogroup utility
 *  \{
 */

/*! \addtogroup tuple
 *  \{
 */

/*! \cond
 */

struct null_type;

/*! \endcond
 */

/*! This metafunction returns the type of a
 *  \p tuple's <tt>N</tt>th element.
 *
 *  \tparam N This parameter selects the element of interest.
 *  \tparam T A \c tuple type of interest.
 *
 *  \see pair
 *  \see tuple
 */
template<int N, class T>
  struct tuple_element
{
  private:
    typedef typename T::tail_type Next;

  public:
    /*! The result of this metafunction is returned in \c type.
     */
    typedef typename tuple_element<N-1, Next>::type type;
}; // end tuple_element

/*! This metafunction returns the number of elements
 *  of a \p tuple type of interest.
 *
 *  \tparam T A \c tuple type of interest.
 *
 *  \see pair
 *  \see tuple
 */
template<class T>
  struct tuple_size
{
  /*! The result of this metafunction is returned in \c value.
   */
  static const int value = 1 + tuple_size<typename T::tail_type>::value;
}; // end tuple_size

// get function for non-const cons-lists, returns a reference to the element

/*! The \p get function returns a reference to a \p tuple element of
 *  interest.
 *
 *  \param t A reference to a \p tuple of interest.
 *  \return A reference to \p t's <tt>N</tt>th element.
 *
 *  \tparam N The index of the element of interest.
 *
 *  The following code snippet demonstrates how to use \p get to print
 *  the value of a \p tuple element.
 *
 *  \code
 *  #include <thrust/tuple.h>
 *  #include <iostream>
 *  ...
 *  thrust::tuple<int, const char *> t(13, "thrust");
 *
 *  std::cout << "The 1st value of t is " << thrust::get<0>(t) << std::endl;
 *  \endcode
 *
 *  \see pair
 *  \see tuple
 */
template<int N, class HT, class TT>
__host__ __device__
inline typename access_traits<
                  typename tuple_element<N, detail::cons<HT, TT> >::type
                >::non_const_type
get(detail::cons<HT, TT>& t);


/*! The \p get function returns a \c const reference to a \p tuple element of
 *  interest.
 *
 *  \param t A reference to a \p tuple of interest.
 *  \return A \c const reference to \p t's <tt>N</tt>th element.
 *
 *  \tparam N The index of the element of interest.
 *
 *  The following code snippet demonstrates how to use \p get to print
 *  the value of a \p tuple element.
 *
 *  \code
 *  #include <thrust/tuple.h>
 *  #include <iostream>
 *  ...
 *  thrust::tuple<int, const char *> t(13, "thrust");
 *
 *  std::cout << "The 1st value of t is " << thrust::get<0>(t) << std::endl;
 *  \endcode
 *
 *  \see pair
 *  \see tuple
 */
template<int N, class HT, class TT>
__host__ __device__
inline typename access_traits<
                  typename tuple_element<N, detail::cons<HT, TT> >::type
                >::const_type
get(const detail::cons<HT, TT>& t);



/*! \p tuple is a class template that can be instantiated with up to ten arguments.
 *  Each template argument specifies the type of element in the \p tuple.
 *  Consequently, tuples are heterogeneous, fixed-size collections of values. An
 *  instantiation of \p tuple with two arguments is similar to an instantiation
 *  of \p pair with the same two arguments. Individual elements of a \p tuple may
 *  be accessed with the \p get function.
 *
 *  \tparam TN The type of the <tt>N</tt> \c tuple element. Thrust's \p tuple
 *          type currently supports up to ten elements.
 *
 *  The following code snippet demonstrates how to create a new \p tuple object
 *  and inspect and modify the value of its elements.
 *
 *  \code
 *  #include <thrust/tuple.h>
 *  #include <iostream>
 *  ...
 *  // create a tuple containing an int, a float, and a string
 *  thrust::tuple<int, float, const char*> t(13, 0.1f, "thrust");
 *
 *  // individual members are accessed with the free function get
 *  std::cout << "The first element's value is " << thrust::get<0>(t) << std::endl; 
 *
 *  // or the member function get
 *  std::cout << "The second element's value is " << t.get<1>() << std::endl;
 *
 *  // we can also modify elements with the same function
 *  thrust::get<0>(t) += 10;
 *  \endcode
 *
 *  \see pair
 *  \see get
 *  \see make_tuple
 *  \see tuple_element
 *  \see tuple_size
 *  \see tie
 */
template <class... Ts>
  class tuple :
    public detail::map_tuple_to_cons<Ts...>::type
{
  /*! \cond
   */

  private:
  typedef typename detail::map_tuple_to_cons<Ts...>::type inherited;

  /*! \endcond
   */

  public:
  /*! \p tuple's no-argument constructor initializes each element.
   */
  inline __host__ __device__
  tuple(void) {}

  /*! \p tuple's constructor copy constructs the elements from the given parameters
   *  \param ts The value to assign to this \p tuple's elements.
   */
  inline __host__ __device__
  tuple(typename access_traits<Ts>::parameter_type... ts)
    : inherited(ts...) {}


  template<class U1, class U2>
  inline __host__ __device__ 
  tuple(const detail::cons<U1, U2>& p) : inherited(p) {}

  __thrust_exec_check_disable__
  template <class U1, class U2>
  inline __host__ __device__ 
  tuple& operator=(const detail::cons<U1, U2>& k)
  {
    inherited::operator=(k);
    return *this;
  }

  /*! \endcond
   */

  /*! This assignment operator allows assigning the first two elements of this \p tuple from a \p pair.
   *  \param k A \p pair to assign from.
   */
  __thrust_exec_check_disable__
  template <class U1, class U2>
  __host__ __device__ inline
  tuple& operator=(const thrust::pair<U1, U2>& k) {
    //BOOST_STATIC_ASSERT(length<tuple>::value == 2);// check_length = 2
    this->head = k.first;
    this->tail.head = k.second;
    return *this;
  }

  /*! \p swap swaps the elements of two <tt>tuple</tt>s.
   *
   *  \param t The other <tt>tuple</tt> with which to swap.
   */
  inline __host__ __device__
  void swap(tuple &t)
  {
    inherited::swap(t);
  }
};

/*! \cond
 */

template <>
class tuple<> : public null_type
{
public:
  typedef null_type inherited;
};

/*! \endcond
 */


/*! This version of \p make_tuple creates a new \c tuple object from objects.
 *
 *  \param ts The objects to copy from.
 *  \return A \p tuple object with members which are copies of \p ts.
 */
template<class... Ts>
__host__ __device__ inline
  typename detail::make_tuple_mapper<Ts...>::type
    make_tuple(const Ts&... ts);

/*! This version of \p tie creates a new \c tuple of references object which
 *  refers to this function's arguments.
 *
 *  \param ts The objects to reference.
 *  \return A \p tuple object with members which are references to \p ts.
 */
template<typename... Ts>
__host__ __device__ inline
tuple<Ts&...> tie(Ts&... ts);

/*! \p swap swaps the contents of two <tt>tuple</tt>s.
 *
 *  \param x The first \p tuple to swap.
 *  \param y The second \p tuple to swap.
 */
template<
  typename... Ts, typename... Us
>
inline __host__ __device__
void swap(tuple<Ts...> &x,
          tuple<Us...> &y);



/*! \cond
 */

__host__ __device__ inline
bool operator==(const null_type&, const null_type&);

__host__ __device__ inline
bool operator>=(const null_type&, const null_type&);

__host__ __device__ inline
bool operator<=(const null_type&, const null_type&);

__host__ __device__ inline
bool operator!=(const null_type&, const null_type&);

__host__ __device__ inline
bool operator<(const null_type&, const null_type&);

__host__ __device__ inline
bool operator>(const null_type&, const null_type&);

/*! \endcond
 */

/*! \} // tuple
 */

/*! \} // utility
 */

} // end thrust

