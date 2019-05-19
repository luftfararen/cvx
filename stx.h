#if !defined(STX_H)
#define STX_H

#include <chrono>
#include <thread>
#include <mutex>
#include <valarray>

namespace stx {

///StopWath class
struct StopWatch
{
public:
  StopWatch()
      : susupending_duration_(0)
      , suspending_(false)
  {
    pre_ = std::chrono::high_resolution_clock::now();
  }

  ///returns lap time in ms.
  double lap()
  {
    auto tmp = std::chrono::high_resolution_clock::now();  //sotres time.
    auto dur = tmp - pre_;
    if(!suspending_) dur += susupending_duration_;
    pre_ = tmp;
    return std::chrono::duration_cast<std::chrono::nanoseconds>(dur).count() / 1000000.0;
  }
  ///suspends time count
  void suspend()
  {
    susupended_time_ = std::chrono::high_resolution_clock::now();
    suspending_ = true;
  }

  ///resumues time count
  void resumue()
  {
    auto tmp = std::chrono::high_resolution_clock::now();  //sotres time.
    susupending_duration_ = tmp - susupended_time_;
    suspending_ = false;
  }

private:
  std::chrono::high_resolution_clock::time_point pre_;
  std::chrono::high_resolution_clock::time_point susupended_time_;
  std::chrono::high_resolution_clock::duration susupending_duration_;
  bool suspending_;
};


template<class T>
T rdiv(T v, T d)
{
  if(v > 0) {
    if(d > 0) {
      return (v + d / 2) / d;
    } else {
      return (v - d / 2) / d;
    }
  } else {
    if(d > 0) {
      return -(-v + d / 2) / d;
    } else {
      return -(-v - d / 2) / d;
    }
  }
}
float rdiv(float v, float d) { return v / d; }
double rdiv(double v, double d) { return v / d; }

template<class INTEGRAL_TYPE>
constexpr int iround(INTEGRAL_TYPE v)
{
  return static_cast<int>(v);
}

constexpr int iround(float _v)
{
	double v = _v;
  if(v >= 0) {
    return static_cast<int>(v + 0.5f);
  } else {
    return -static_cast<int>(-v + 0.5f);
  }
}

constexpr int iround(double _v)
{
	long double v = _v;
  if(v >= 0) {
    return static_cast<int>(v + 0.5);
  } else {
    return -static_cast<int>(-v + 0.5);
  }
}

inline int iround(long double v)
{
  return static_cast<int>(llround(v));
}

template<class INTEGRAL_TYPE>
constexpr int fast_iround(INTEGRAL_TYPE v)
{
  return static_cast<int>(v);
}

constexpr int fast_iround(float v)
{
  if(v >= 0) {
    return static_cast<int>(v + 0.5f);
  } else {
    return -static_cast<int>(-v + 0.5f);
  }
}

constexpr int fast_iround(double v)
{
  if(v >= 0) {
    return static_cast<int>(v + 0.5);
  } else {
    return -static_cast<int>(-v + 0.5);
  }
}

constexpr int fast_iround(long double v)
{
  if(v >= 0) {
    return static_cast<int>(v + 0.5l);
  } else {
    return -static_cast<int>(-v + 0.5l);
  }
}


template<class INTEGRAL_TYPE>
constexpr int itrunc(INTEGRAL_TYPE v)
{
  return static_cast<int>(v);
}

constexpr int itrunc(float v)
{
  if(v >= 0) {
    return static_cast<int>(v);
  } else {
    return -static_cast<int>(-v);
  }
}

constexpr int itrunc(double v)
{
  if(v >= 0) {
    return static_cast<int>(v);
  } else {
    return -static_cast<int>(-v);
  }
}

constexpr int itrunc(long double v)
{
  if(v >= 0) {
    return static_cast<int>(v);
  } else {
    return -static_cast<int>(-v);
  }
}

}  // namespace stx

#endif  //#if !defined(STX_H)