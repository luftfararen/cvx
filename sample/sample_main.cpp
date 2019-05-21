#include <iostream>
#include <iomanip>
#include <execution>
#include <algorithm>
#include <amp.h>
#include <amp_graphics.h>
#include "cvx_exp.h"
#include "stx.h"
#include "amp_utils.h"

using namespace cvx;
using namespace stx;
/*
using namespace cvx;
template<class FUNC>
void paral_for_idx(int st, int ed, FUNC func)
{
  int   num_cpu = std::thread::hardware_concurrency();
  int nstripes = (ed - st + num_cpu -1) / num_cpu;
  std::cout << nstripes << std::endl;
  cv::parallel_for_(cv::Range(st, ed),
                    [&func](const cv::Range& range) {
    std::lock_guard<std::mutex> lock(mtx);
    std::cout << range.start << " " << range.end-1 << std::endl; 
    for(int idx = range.start; idx < range.end; idx++) {
      func(idx);
    }
  },nstripes);
}
*/

void print_time(const std::string label, double time, const std::string& note = "")
{
  using namespace std;
  cout << "|" << label << "|" << note << "|" << fixed << setprecision(2) << time << "|" << endl;
}

template<class EXPO, class CONT, class T>
void fill_all(EXPO&& expo, CONT& cont, const T& value)
{
  fill(expo, cont.begin(), cont.end(), value);
}

template<class... A>
void test(A... args)
{
  for(int i : std::initializer_list<int>{args...}) {
    std::cout << i << std::endl;
  }
}

void f1(Mtx1b& src, Mtx1b& ref, int loop_count)
{
  using namespace std;
  int h = src.rows;
  int w = src.cols;

  std::string label = "1|画面外座標値をclamp";
  StopWatch sw;
  for(int z = 0; z < loop_count; z++) {
    for(int y = 0; y < h; y++) {
      for(int x = 0; x < w; x++) {
        int v = (4 * src(clamp(y, 0, h - 1), clamp(x, 0, w - 1)) +
                 2 *
                     (src(clamp(y, 0, h - 1), clamp(x - 1, 0, w - 1)) +
                      src(clamp(y - 1, 0, h - 1), clamp(x, 0, w - 1)) +
                      src(clamp(y, 0, h - 1), clamp(x + 1, 0, w - 1)) +
                      src(clamp(y + 1, 0, h - 1), clamp(x, 0, w - 1))) +

                 (src(clamp(y - 1, 0, h - 1), clamp(x - 1, 0, w - 1)) +
                  src(clamp(y - 1, 0, h - 1), clamp(x + 1, 0, w - 1)) +
                  src(clamp(y + 1, 0, h - 1), clamp(x - 1, 0, w - 1)) +
                  src(clamp(y + 1, 0, h - 1), clamp(x + 1, 0, w - 1)))) /
            16;
        ref(y, x) = v;
      }
    }
  }
  print_time(label, sw.lap() / loop_count);
}

void f2(Mtx1b& src, Mtx1b& dst, int loop_count)
{
  using namespace std;
  int h = src.rows;
  int w = src.cols;

  std::string label = "2|border付き画像の利用";
  StopWatch sw;
  for(int z = 0; z < loop_count; z++) {
    src.extrapolate();
    for(int y = 0; y < h; y++) {
      for(int x = 0; x < w; x++) {
        dst(y, x) =
            (4 * src(y, x) + 2 * (src(y, x - 1) + src(y - 1, x) + src(y, x + 1) + src(y + 1, x)) +

             (src(y - 1, x - 1) + src(y - 1, x + 1) + src(y + 1, x + 1) + src(y + 1, x - 1))) /
            16;
      }
    }
  }
  print_time(label, sw.lap() / loop_count, "b");
}

void f3(Mtx1b& src, Mtx1b& dst, int loop_count)
{
  using namespace std;
  int h = src.rows;
  int w = src.cols;

  std::string label = "3|case2+二次元アクセス可能な画素値用ポインタークラスの利用 ";
  StopWatch sw;
  for(int z = 0; z < loop_count; z++) {
    src.extrapolate();
    for(int y = 0; y < h; y++) {
      auto pp = src.createPixPtr(y, 0);
      for(int x = 0; x < w; x++) {
        dst(y, x) = (4 * pp(0, 0) + 2 * (pp(0, -1) + pp(-1, 0) + pp(0, 1) + pp(1, 0)) +
                     (pp(-1, -1) + pp(-1, 1) + pp(1, 1) + pp(1, -1))) /
            16;
        pp++;
      }
    }
  }
  print_time(label, sw.lap() / loop_count, "b");
}

void f4(Mtx1b& src, Mtx1b& dst, int loop_count)
{
  using namespace std;
  int h = src.rows;
  int w = src.cols;

  std::string label = "4|case3でループを展開";
  StopWatch sw;
  for(int z = 0; z < loop_count; z++) {
    src.extrapolate();
    for(int y = 0; y < h; y++) {
      auto pp = src.createPixPtr(y, 0);
      for(int x = 0; x < w; x++) {
        dst(y, x) = (pp(-1, -1) + 2 * pp(-1, 0) + pp(-1, 1) + 2 * pp(0, -1) + 4 * pp(0, 0) +
                     2 * pp(0, 1) + pp(1, -1) + 2 * pp(1, 0) + pp(1, 1)) /
            16;
        pp++;
      }
    }
  }
  print_time(label, sw.lap() / loop_count, "b");
}

void f5(Mtx1b& src, Mtx1b& dst, int loop_count)
{
  using namespace std;
  int h = src.rows;
  int w = src.cols;

  std::string label = "5|case4でopenmpをさらに利用";
  StopWatch sw;
  for(int z = 0; z < loop_count; z++) {
    src.extrapolate();
    paral_for_pp_(src, dst, [](const  PixPtr<uchar>& pp) {  //OpenMP利用
      return (pp(-1, -1) + 2 * pp(-1, 0) + pp(-1, 1) + 2 * pp(0, -1) + 4 * pp(0, 0) + 2 * pp(0, 1) +
              pp(1, -1) + 2 * pp(1, 0) + pp(1, 1)) /
          16;
    });
  }
  print_time(label, sw.lap() / loop_count, "bp");
}

void f6(Mtx1b& src, Mtx1b& dst, int loop_count)
{
  using namespace std;
  int h = src.rows;
  int w = src.cols;

  Kernel<int, 1> ker(1, 2, 1, 2, 4, 2, 1, 2, 1);

  std::string label = "6|自作Kernelクラスの利用";
  StopWatch sw;
  for(int z = 0; z < loop_count; z++) {
    src.extrapolate();
    paral_for_pp_(src, dst, [&](const auto& pp) {
#if 1
      int sum = 0;
      for_dydx(ker.radius, ker.radius, [&](int dy, int dx) { sum += pp(dy, dx) * ker(dy, dx); });
      return sum / 16;
#else
      return ker.convolute(pp) / 16;
#endif
    });
  }
  print_time(label, sw.lap() / loop_count, "bp");
}

void f7(Mtx1b& src, Mtx1b& dst, int loop_count)
{
  using namespace std;
  int h = src.rows;
  int w = src.cols;

  Kernel<int, 1> ker(1, 2, 1, 2, 4, 2, 1, 2, 1);

  std::string label = "7|case5をベースにでauto vectrizationしやすいループにする";
  StopWatch sw;
  for(int z = 0; z < loop_count; z++) {
    src.extrapolate();

#pragma omp parallel for
    for(int y = 0; y < h; y++) {
      auto pd = dst[y];
      int sum[4096] = {};
      //for(int i=0;i<sum.size();i++) sum[i] = 0;
      for_dydx(ker.radius, ker.radius, [&](int dy, int dx) {
        int k = ker(dy, dx);
        auto ps = src[0] + src.stepT() * (dy + y) + dx;
        for_ps_pd_idx_old<32>(ps, sum, w,
                              [&](auto& ps, auto& psum, int d) { psum[d] += k * ps[d]; });
      });
      for_ps_pd_idx_old<32>(sum, pd, w, [&](auto& psum, auto& pd, int d) { pd[d] = psum[d] / 16; });
    }
  }
  print_time(label, sw.lap() / loop_count, "bp");
}

void f7_1(Mtx1b& src, Mtx1b& dst, int loop_count)
{
  std::string label = "7.1|case7をベースに少し読みやすくする";

  using namespace std;
  int h = src.rows;
  int w = src.cols;

  const Kernel<int, 1> ker(1, 2, 1, 2, 4, 2, 1, 2, 1);

  StopWatch sw;
  for(int z = 0; z < loop_count; z++) {
    src.extrapolate();
#pragma omp parallel for
    for(int y = 0; y < h; y++) {
      auto pd = dst[y];
      auto pp = src.createPixPtr(y, 0);
      int sum[4096] = {};
      for_dydx(ker.radius, ker.radius, [&](int dy, int dx) {
        int k = ker(dy, dx);
        auto ps = pp.ptr(dy, dx);
        for_ps_pd_idx_old<32>(ps, sum, w,
                              [&](auto& ps, auto& psum, int d) { psum[d] += k * ps[d]; });
      });
      for_ps_pd_idx_old<32>(sum, pd, w, [&](auto& psum, auto& pd, int d) { pd[d] = psum[d] / 16; });
    }
  }
  print_time(label, sw.lap() / loop_count, "bp");
}

void f7_2(Mtx1b& src, Mtx1b& dst, int loop_count)
{
  std::string label =
      "7.2|case7をベースに自作カーネルクラスの利用をやめる,auto vectrizationしやすいループをやめる";

  using namespace std;
  int h = src.rows;
  int w = src.cols;

  //  const Kernel<int, 1> ker(1, 2, 1, 2, 4, 2, 1, 2, 1);
  static const int ker[] = {1, 2, 1, 2, 4, 2, 1, 2, 1};

  StopWatch sw;
  for(int z = 0; z < loop_count; z++) {
    src.extrapolate();
#pragma omp parallel for
    for(int y = 0; y < h; y++) {
      auto pd = dst[y];
      auto pp = src.createPixPtr(y, 0);
      int sum[4096] = {};
      int j = 0;
      for_dydx(1, 1, [&](int dy, int dx) {
        //int k = ker(dy, dx);
        int k = ker[j];
        auto ps = pp.ptr(dy, dx);
        for(int x = 0; x < w; x++) {
          sum[x] += k * ps[x];
        }
        j++;
      });
      for(int x = 0; x < w; x++) {
        pd[x] = sum[x] / 16;
      }
    }
  }
  print_time(label, sw.lap() / loop_count, "bp");
  //  print(dst);
}

void f7_3(Mtx1b& src, Mtx1b& dst, int loop_count)
{
  using namespace std;
  int h = src.rows;
  int w = src.cols;

  static const int ker[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};

  std::string label = "7.3|case7で自作カーネルクラスの利用をやめる";
  StopWatch sw;
  for(int z = 0; z < loop_count; z++) {
    src.extrapolate();
#pragma omp parallel for
    for(int y = 0; y < h; y++) {
      auto pd = dst[y];
      int sum[4096] = {};
      int j = 0;
      for_dydx(1, 1, [&](int dy, int dx) {
        int k = ker[j];
        auto ps = src[0] + src.stepT() * (dy + y) + dx;
        for_ps_pd_idx_old<32>(ps, sum, w,
                              [&](auto& ps, auto& psum, int d) { psum[d] += k * ps[d]; });
        j++;
      });
      for_ps_pd_idx_old<32>(sum, pd, w, [&](auto& psum, auto& pd, int d) { pd[d] = psum[d] / 16; });
    }
  }
  print_time(label, sw.lap() / loop_count, "bp");
}

void f7_4(Mtx1b& src, Mtx1b& dst, int loop_count)
{
  std::string label = "7.4|case7.1をベースにループのかけ方をかえる";

  using namespace std;
  int h = src.rows;
  int w = src.cols;

  static constexpr int UNIT = 64;

  const Kernel<int, 1> ker(1, 2, 1, 2, 4, 2, 1, 2, 1);

  StopWatch sw;
  for(int z = 0; z < loop_count; z++) {
    src.extrapolate();
#pragma omp parallel for
    for(int y = 0; y < h; y++) {
      auto pd = dst[y];
      auto pp = src.createPixPtr(y, 0);
      alignas(32) int sum[4096] = {};
      for_dydx(1, 1, [&](int dy, int dx) {
        const int k = ker(dy, dx);
        const auto ps = pp.ptr(dy, dx);
        simd_for_n<UNIT>(ps, sum, w, [&k](const auto& s, auto& d) { d += k * s; });
      });
      simd_for_n<UNIT>(sum, pd, w, [](const auto& s, auto& d) { d = s / 16; });
    }
  }
  print_time(label, sw.lap() / loop_count, "bp");
}

//void f7_5(Mtx1b& src, Mtx1b& dst, int loop_count)
//{
//  std::string label = "7.5|case7.1をベースにループのかけ方をかえる";
//
//  using namespace std;
//  int h = src.rows;
//  int w = src.cols;
//
//  static constexpr int UNIT = 64;
//
//  const Kernel<int, 1> ker(1, 2, 1, 2, 4, 2, 1, 2, 1);
//
//  StopWatch sw;
//  for(int z = 0; z < loop_count; z++) {
//    src.extrapolate();
//    Mtx1i tmp(src.size());
//    for_dy_dx(1, 1, [&](int dy, int dx) {
//      const int k = ker(dy, dx);
//      paral_for_pp_offset(src, [&](const auto& pps) {
//				return k * pps(dy, dx); 
//			});
//    });
//    paral_for_pp_(tmp, dst, [&](const auto& pps) { return pps(0, 0) / 16; });
//  }
//  print_time(label, sw.lap() / loop_count, "bp");
//}

void f9(Mtx1b& src, Mtx1b& dst, int loop_count)
{
  using namespace std;
  int h = src.rows;
  int w = src.cols;

  SepKernel<int, 1> sker(2, 1);

  StopWatch sw;
  std::string label = "9|分離型フィルタ";
  for(int z = 0; z < loop_count; z++) {
    src.extrapolate();
    auto tmp = Mtx1i::createWithBorder(src.size(), sker.radius);
    paral_for_pp_(src, tmp, [&](const auto& pp) { return sker.convolute(pp, false); });
    tmp.extrapolate();
    paral_for_pp_(tmp, dst, [&](const auto& pp) { return sker.convolute(pp, true) / 16; });
  }
  print_time(label, sw.lap() / loop_count, "bp");
}

void f10(Mtx1b& src, Mtx1b& dst, int loop_count)
{
  using namespace std;
  int h = src.rows;
  int w = src.cols;

  SepKernel<int, 1> sker(2, 1);

  std::string label = "10|分離型フィルタで	auto vectorizationしやすいループ ";
  StopWatch sw;
  //SepKernel<int, 1> sker(2, 1);
  for(int z = 0; z < loop_count; z++) {
    src.extrapolate();
    auto tmp = Mtx1i::createWithBorder(src.size(), sker.radius);
#pragma omp parallel for
    for(int y = 0; y < h; y++) {
      auto ptl = tmp[y];
      for_p_idx<32>(w, ptl, [&](auto& pt, int d) { pt[d] = 0; });
      for(int dy = -sker.radius; dy <= sker.radius; dy++) {
        auto k = sker(dy);
        auto psl = src[0] + src.stepT() * (dy + y);
        for_ps_pd_idx_old<32>(psl, ptl, w, [&](auto& ps, auto& pt, int d) { pt[d] += k * ps[d]; });
      }
    }
    tmp.extrapolate();
#pragma omp parallel for
    for(int y = 0; y < h; y++) {
      int sum[4096] = {0};
      for(int dx = -sker.radius; dx <= sker.radius; dx++) {
        auto k = sker(dx);
        auto ptl = tmp[y] + dx;
        for_ps_pd_idx_old<32>(ptl, sum, w, [&](auto& pt, auto& pd, int d) { pd[d] += k * pt[d]; });
      }
      for_ps_pd_idx_old<32>(sum, dst[y], w, [&](auto& pt, auto& pd, int d) { pd[d] = pt[d] / 16; });
    }
  }
  print_time(label, sw.lap() / loop_count, "bp");
}

void f10_1(Mtx1b& src, Mtx1b& dst, int loop_count)
{
  using namespace std;
  int h = src.rows;
  int w = src.cols;

  SepKernel<int, 1> sker(2, 1);

  std::string label = "10.1|分離型フィルタで	auto vectorizationしやすいループ、さらに最適化";
  StopWatch sw;
  for(int z = 0; z < loop_count; z++) {
    src.extrapolate();
#pragma omp parallel for
    for(int y = 0; y < h; y++) {
      auto pps = src.createPixPtr(y, 0);
      alignas(32) int vbuf_entity[4096] = {};  //cv::AutoBufferにすべきか？　
      int* vbuf = vbuf_entity + 32;
      //for_p_idx<32>(w, ptl, [&](auto& pt, int d) { pt[d] = 0; });
      for(int dy = -sker.radius; dy <= sker.radius; dy++) {
        const auto k = sker(dy);
        const auto psl = pps.ptr(dy, 0);
        for_ps_pd_idx_old<32>(psl, vbuf, w,
                              [k](const auto& ps, const auto& pt, int d) { pt[d] += k * ps[d]; });
      }
      vbuf[-1] = vbuf[0];
      vbuf[w] = vbuf[w - 1];
      alignas(32) int bufh[4096] = {};  //cv::AutoBufferにすべきか？
      for(int dx = -sker.radius; dx <= sker.radius; dx++) {
        const auto k = sker(dx);
        const auto ptl = vbuf + dx;
        for_ps_pd_idx_old<32>(ptl, bufh, w,
                              [&](const auto& pt, auto& pd, int d) { pd[d] += k * pt[d]; });
      }
      for_ps_pd_idx_old<32>(bufh, dst[y], w,
                            [](const auto& pt, auto& pd, int d) { pd[d] = pt[d] / 16; });
    }
  }
  print_time(label, sw.lap() / loop_count, "bp");
}

void f10_2(Mtx1b& src, Mtx1b& dst, int loop_count)
{
  using namespace std;
  int h = src.rows;
  int w = src.cols;

  constexpr int UNIT = 64;
  SepKernel<int, 1> sker(2, 1);

  std::string label = "10.2|分離型フィルタで	auto vectorizationしやすいループ、さらにさらに最適化";
  StopWatch sw;
  for(int z = 0; z < loop_count; z++) {
    src.extrapolate();
    //auto tmp = Mtx1i::createWithBorder(src.size(), sker.radius);
#pragma omp parallel for
    for(int y = 0; y < h; y++) {
      auto pps = src.createPixPtr(y, 0);
      alignas(32) int vbuf_entity[4096] = {};  //cv::AutoBufferにすべきか？　
      int* vbuf = vbuf_entity + 32;
      for(int dy = -sker.radius; dy <= sker.radius; dy++) {
        const auto k = sker(dy);
        const auto psl = pps.ptr(dy, 0);
        //参照キャプチャ[&k]でなく、コピーキャプチャ[k]にすると遅い、SIMD化されな？
        for_ps_pd_idx<UNIT>(psl, vbuf, w,
                        [&k](const auto& ps, auto& pt, int d) { pt[d] += k * ps[d]; });
      }
      vbuf[-1] = vbuf[0];
      vbuf[w] = vbuf[w - 1];
      alignas(32) int bufh[4096] = {};  //cv::AutoBufferにすべきか？
      for(int dx = -sker.radius; dx <= sker.radius; dx++) {
        const auto k = sker(dx);
        const auto ptl = vbuf + dx;
        for_ps_pd_idx<UNIT>(ptl, bufh, w,
                        [&k](const auto& pt, auto& pd, int d) { pd[d] += k * pt[d]; });
      }
      for_ps_pd_idx<UNIT>(bufh, dst[y], w,
                          [](const auto& pt, auto& pd, int d) { pd[d] = pt[d] / 16; });
    }
  }
  print_time(label, sw.lap() / loop_count, "bp");
}

void f10_3(Mtx1b& src, Mtx1b& dst, int loop_count)
{
  using namespace std;
  int h = src.rows;
  int w = src.cols;

  constexpr int UNIT = 32;

  SepKernel<int, 1> sker(2, 1);

  std::string label = "10.3|case10.2でループの書き方を少し変える(32pix単位)";

  StopWatch sw;
  for(int z = 0; z < loop_count; z++) {
    src.extrapolate();
    //auto tmp = Mtx1i::createWithBorder(src.size(), sker.radius);
#pragma omp parallel for
    for(int y = 0; y < h; y++) {
      auto pps = src.createPixPtr(y, 0);
      alignas(32) int vbuf_entity[4096] = {};  //cv::AutoBufferにすべきか？　
      int* vbuf = vbuf_entity + 32;
      for(int dy = -sker.radius; dy <= sker.radius; dy++) {
        const auto k = sker(dy);
        const auto psl = pps.ptr(dy, 0);
        simd_for_n<UNIT>(psl, vbuf, w, [&k](const auto& s, auto& d) { d += k * s; });
      }
      vbuf[-1] = vbuf[0];
      vbuf[w] = vbuf[w - 1];
      alignas(32) int bufh[4096] = {};  //cv::AutoBufferにすべきか？
      for(int dx = -sker.radius; dx <= sker.radius; dx++) {
        const auto k = sker(dx);
        const auto ptl = vbuf + dx;
        simd_for_n<UNIT>(ptl, bufh, w, [&k](const auto& s, auto& d) { d += k * s; });
      }
      simd_for_n<UNIT>(bufh, dst[y], w, [&](const auto& s, auto& d) { d = s / 16; });
    }
  }
  print_time(label, sw.lap() / loop_count, "bp");
}

void f10_4(Mtx1b& src, Mtx1b& dst, int loop_count)
{
  using namespace std;
  int h = src.rows;
  int w = src.cols;

  constexpr int UNIT = 64;

  SepKernel<int, 1> sker(2, 1);

  std::string label = "10.4|case10.3で64pix単位にする";

  StopWatch sw;
  for(int z = 0; z < loop_count; z++) {
    src.extrapolate();
    //auto tmp = Mtx1i::createWithBorder(src.size(), sker.radius);
#pragma omp parallel for
    for(int y = 0; y < h; y++) {
      auto pps = src.createPixPtr(y, 0);
      alignas(32) int vbuf_entity[4096] = {};  //cv::AutoBufferにすべきか？　
      int* vbuf = vbuf_entity + 32;
      for(int dy = -sker.radius; dy <= sker.radius; dy++) {
        const auto k = sker(dy);
        const auto psl = pps.ptr(dy, 0);
        simd_for_n<UNIT>(psl, vbuf, w, [&k](const auto& s, auto& d) { d += k * s; });
      }
      vbuf[-1] = vbuf[0];
      vbuf[w] = vbuf[w - 1];
      alignas(32) int bufh[4096] = {};  //cv::AutoBufferにすべきか？
      for(int dx = -sker.radius; dx <= sker.radius; dx++) {
        const auto k = sker(dx);
        const auto ptl = vbuf + dx;
        simd_for_n<UNIT>(ptl, bufh, w, [&k](const auto& s, auto& d) { d += k * s; });
      }
      simd_for_n<UNIT>(bufh, dst[y], w, [&](const auto& s, auto& d) { d = s / 16; });
    }
  }
  print_time(label, sw.lap() / loop_count, "bp");
}

void f10_5(Mtx1b& src, Mtx1b& dst, int loop_count)
{
  using namespace std;
  int h = src.rows;
  int w = src.cols;

  constexpr int UNIT = 16;

  SepKernel<int, 1> sker(2, 1);

  std::string label = "10.5|case10.3で16pix単位にする";

  StopWatch sw;
  for(int z = 0; z < loop_count; z++) {
    src.extrapolate();
    //auto tmp = Mtx1i::createWithBorder(src.size(), sker.radius);
#pragma omp parallel for
    for(int y = 0; y < h; y++) {
      auto pps = src.createPixPtr(y, 0);
      alignas(32) int vbuf_entity[4096] = {};  //cv::AutoBufferにすべきか？　
      int* vbuf = vbuf_entity + 32;
      for(int dy = -sker.radius; dy <= sker.radius; dy++) {
        const auto k = sker(dy);
        const auto psl = pps.ptr(dy, 0);
        simd_for_n<UNIT>(psl, vbuf, w, [&k](const auto& s, auto& d) { d += k * s; });
      }
      vbuf[-1] = vbuf[0];
      vbuf[w] = vbuf[w - 1];
      alignas(32) int bufh[4096] = {};  //cv::AutoBufferにすべきか？
      for(int dx = -sker.radius; dx <= sker.radius; dx++) {
        const auto k = sker(dx);
        const auto ptl = vbuf + dx;
        simd_for_n<UNIT>(ptl, bufh, w, [&k](const auto& s, auto& d) { d += k * s; });
      }
      simd_for_n<UNIT>(bufh, dst[y], w, [&](const auto& s, auto& d) { d = s / 16; });
    }
  }
  print_time(label, sw.lap() / loop_count, "bp");
}

void f11(Mtx1b& src, Mtx1b& dst, int loop_count)
{
  using namespace std;
  using namespace Concurrency;
  using namespace Concurrency::graphics;
  namespace cc = Concurrency;
  namespace cg = Concurrency::graphics;

  int h = src.rows;
  int w = src.cols;

  std::string label = "11|C++AMP byte ";

  //わかりやすさのためにborder抜きの画像を作る、時間計測からは除外
  Mtx1b tmp(h, w);
  paral_for_pp_(src, tmp, [&](const auto& pp) { return pp(0, 0); });

  StopWatch sw;
  for(int z = 0; z < loop_count; z++) {
    cc::extent<2> ext(h, w);
    texture<uint, 2> stex(ext, tmp[0], h * w, 8U);
    texture_view<const uint, 2> sview(stex);
    texture<uint, 2> dtex(ext, 8U);

    parallel_for_each(ext, [&, sview, h, w](cc::index<2> idx) restrict(amp) {
      int sum = 0;
      int j = 0;
      int karr[] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
      for(int dy = -1; dy <= 1; dy++) {
        for(int dx = -1; dx <= 1; dx++) {
          cc::index<2> idx1;
          idx1[0] = cc::clamp(idx[0] + dy, 0, h - 1);
          idx1[1] = cc::clamp(idx[1] + dx, 0, w - 1);
          sum += karr[j] * sview[idx1];
          j++;
        }
      }
      dtex.set(idx, sum / 16);
    });
    cg::copy(dtex, dst[0], h * w);
  }
  print_time(label, sw.lap() / loop_count);
}

#if defined(_MSC_VER)

void f12(Mtx1b& src, Mtx1b& dst, int loop_count)
{
  using namespace std;
  int h = src.rows;
  int w = src.cols;

  std::string label = "12|C++AMP int";

  using namespace Concurrency;
  using namespace Concurrency::graphics;
  namespace cc = Concurrency;
  namespace cg = Concurrency::graphics;

  Mtx1i tmp(h, w);
  Mtx1i tmp2(h, w);

  StopWatch sw;
  for(int z = 0; z < loop_count; z++) {
    paral_for_pp_(src, tmp, [&](const auto& pp) { return pp(0, 0); });

    cc::extent<2> ext(h, w);
    texture<uint, 2> stex(ext, tmp[0], h * w * 4, 32);
    texture_view<const uint, 2> sview(stex);
    texture<uint, 2> dtex(ext, 32);

    parallel_for_each(ext, [&, sview, h, w](cc::index<2> idx) restrict(amp) {
      int sum = 0;
      int j = 0;
      int karr[] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
      for(int dy = -1; dy <= 1; dy++) {
        for(int dx = -1; dx <= 1; dx++) {
          cc::index<2> idx1;
          idx1[0] = cc::clamp(idx[0] + dy, 0, h - 1);
          idx1[1] = cc::clamp(idx[1] + dx, 0, w - 1);
          sum += karr[j] * sview[idx1];
          j++;
        }
      }
      dtex.set(idx, sum / 16);
    });
    cg::copy(dtex, tmp2[0], h * w * 4);
    paral_for_pp_(tmp2, dst, [&](const auto& pp) { return pp(0, 0); });
  }
  print_time(label, sw.lap() / loop_count);
}

void f13(Mtx1b& src, Mtx1b& dst, int loop_count)
{
  using namespace std;
  using namespace Concurrency;
  using namespace Concurrency::graphics;
  namespace cc = Concurrency;
  namespace cg = Concurrency::graphics;

  int h = src.rows;
  int w = src.cols;

  std::string label = "13|C++AMP texture_view<const float, 2>";

  StopWatch sw;
  Mtx1f tmp(h, w);
  Mtx1f tmp2(h, w);
  for(int z = 0; z < loop_count; z++) {
    paral_for_pp_(src, tmp, [&](const auto& pp) { return pp(0, 0); });
    cc::extent<2> ext(h, w);
    texture<float, 2> stex(ext, tmp[0], h * w * 4, 32);
    texture_view<const float, 2> sview(stex);
    texture<float, 2> dtex(ext, 32);

    parallel_for_each(ext, [&, sview, h, w](cc::index<2> idx) restrict(amp) {
      float sum = 0;
      int j = 0;
      const int karr[] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
      for(int dy = -1; dy <= 1; dy++) {
        for(int dx = -1; dx <= 1; dx++) {
          cc::index<2> idx1;
          idx1[0] = cc::clamp(idx[0] + dy, 0, h - 1);
          idx1[1] = cc::clamp(idx[1] + dx, 0, w - 1);
          sum += karr[j] * sview[idx1];
          j++;
        }
      }
      dtex.set(idx, sum / 16);
    });
    cg::copy(dtex, tmp2[0], h * w * 4);
    paral_for_pp_(tmp2, dst, [&](const auto& pp) { return pp(0, 0); });
  }
  print_time(label, sw.lap() / loop_count);
}

void f13_1(Mtx1b& src, Mtx1b& dst, int loop_count)
{
  using namespace std;
  using namespace Concurrency;
  using namespace Concurrency::graphics;
  namespace cc = Concurrency;
  namespace cg = Concurrency::graphics;

  int h = src.rows;
  int w = src.cols;

  std::string label = "13.1|C++AMP array_view<float,2>";

  StopWatch sw;
  Mtx1f tmp(h, w);
  Mtx1f tmp2(h, w);

  for(int z = 0; z < loop_count; z++) {
    paral_for_pp_(src, tmp, [&](const auto& pp) { return pp(0, 0); });
    cc::extent<2> ext(h, w);
    array_view<float, 2> sview(ext, tmp[0]);
    array_view<float, 2> dview(ext, tmp2[0]);
    parallel_for_each(ext, [&, sview, dview, h, w](cc::index<2> idx) restrict(amp) {
      float sum = 0;
      int j = 0;
      const int karr[] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
      for(int dy = -1; dy <= 1; dy++) {
        for(int dx = -1; dx <= 1; dx++) {
          cc::index<2> idx1;
          idx1[0] = cc::clamp(idx[0] + dy, 0, h - 1);
          idx1[1] = cc::clamp(idx[1] + dx, 0, w - 1);
          sum += karr[j] * sview[idx1];
          j++;
        }
      }
      dview[idx] = sum / 16;
    });
    paral_for_pp_(tmp2, dst, [&](const auto& pp) { return pp(0, 0); });
  }
  print_time(label, sw.lap() / loop_count);
}

void f13_2(Mtx1b& src, Mtx1b& dst, int loop_count)
{
  using namespace std;
  using namespace Concurrency;
  using namespace Concurrency::graphics;
  namespace cc = Concurrency;
  namespace cg = Concurrency::graphics;

  int h = src.rows;
  int w = src.cols;

  std::string label = "13.2|C++AMP texture_view<const float, 2> using sample";

  StopWatch sw;
  Mtx1f tmp(h, w);
  Mtx1f tmp2(h, w);
  for(int z = 0; z < loop_count; z++) {
    paral_for_pp_(src, tmp, [&](const auto& pp) { return pp(0, 0); });
    cc::extent<2> ext(h, w);
    texture<float, 2> stex(ext, tmp[0], h * w * 4, 32);
    const texture_view<const float, 2> sview(stex);
    texture<float, 2> dtex(ext, 32);

    parallel_for_each(ext, [&, sview, h, w, ext](cc::index<2> idx) restrict(amp) {
      float sum = 0;
      int j = 0;
      const int karr[] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
      for(int dy = -1; dy <= 1; dy++) {
        for(int dx = -1; dx <= 1; dx++) {
          sum += karr[j] * sview.sample(coord(idx[0] + dx, idx[1] + dy, ext));
          j++;
        }
      }
      dtex.set(idx, sum / 16);
    });
    cg::copy(dtex, tmp2[0], h * w * 4);
    paral_for_pp_(tmp2, dst, [&](const auto& pp) { return uchar(pp(0, 0)); });
  }
  print_time(label, sw.lap() / loop_count);
}

void f14(Mtx1b& src, Mtx1b& dst, int loop_count)
{
  using namespace std;
  using namespace Concurrency;
  using namespace Concurrency::graphics;
  namespace cc = Concurrency;
  namespace cg = Concurrency::graphics;

  int h = src.rows;
  int w = src.cols;

  std::string label = "14|C++AMP texture_view<const float, 2>";

  Mtx1f tmp(h, w);
  Mtx1f tmp2(h, w);
  paral_for_pp_(src, tmp, [&](const auto& pp) { return pp(0, 0); });
  StopWatch sw;
  for(int z = 0; z < loop_count; z++) {
    cc::extent<2> ext(h, w);
    texture<float, 2> stex(ext, tmp[0], h * w * 4, 32);
    texture_view<const float, 2> sview(stex);
    texture<float, 2> dtex(ext, 32);

    parallel_for_each(ext, [&, sview, h, w](cc::index<2> idx) restrict(amp) {
      float sum = 0;
      int j = 0;
      const int karr[] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
      for(int dy = -1; dy <= 1; dy++) {
        for(int dx = -1; dx <= 1; dx++) {
          cc::index<2> idx1;
          idx1[0] = cc::clamp(idx[0] + dy, 0, h - 1);
          idx1[1] = cc::clamp(idx[1] + dx, 0, w - 1);
          sum += karr[j] * sview[idx1];
          j++;
        }
      }
      dtex.set(idx, sum / 16);
    });
    cg::copy(dtex, tmp2[0], h * w * 4);
  }
  print_time(label, sw.lap() / loop_count);
  paral_for_pp_(tmp2, dst, [&](const auto& pp) { return pp(0, 0); });
}

void f14_1(Mtx1b& src, Mtx1b& dst, int loop_count)
{
  std::string label = "14.1|C++AMP array_view<float, 2>";

  using namespace std;
  using namespace Concurrency;
  using namespace Concurrency::graphics;
  namespace cc = Concurrency;
  namespace cg = Concurrency::graphics;

  int h = src.rows;
  int w = src.cols;

  Mtx1f tmp(h, w);
  Mtx1f tmp2(h, w);
  paral_for_pp_(src, tmp, [&](const auto& pp) { return pp(0, 0); });
  StopWatch sw;
  for(int z = 0; z < loop_count; z++) {
    cc::extent<2> ext(h, w);
    array_view<float, 2> sview(ext, tmp[0]);
    array_view<float, 2> dview(ext, tmp2[0]);
    parallel_for_each(ext, [&, sview, dview, h, w](cc::index<2> idx) restrict(amp) {
      float sum = 0;
      int j = 0;
      const int karr[] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
      for(int dy = -1; dy <= 1; dy++) {
        for(int dx = -1; dx <= 1; dx++) {
          cc::index<2> idx1;
          idx1[0] = cc::clamp(idx[0] + dy, 0, h - 1);
          idx1[1] = cc::clamp(idx[1] + dx, 0, w - 1);
          sum += karr[j] * sview[idx1];
          j++;
        }
      }
      dview[idx] = sum / 16;
    });
  }
  paral_for_pp_(tmp2, dst, [&](const auto& pp) { return pp(0, 0); });
  print_time(label, sw.lap() / loop_count);
}

void f14_2(Mtx1b& src, Mtx1b& dst, int loop_count)
{
  std::string label = "14.2|C++AMP texture_view<const float, 2> using sample";

  using namespace std;
  using namespace Concurrency;
  using namespace Concurrency::graphics;
  namespace cc = Concurrency;
  namespace cg = Concurrency::graphics;

  int h = src.rows;
  int w = src.cols;

  Mtx1f tmp(h, w);
  Mtx1f tmp2(h, w);
  paral_for_pp_(src, tmp, [&](const PixPtr<uchar>& pp) { return pp(0, 0); });
  StopWatch sw;
  for(int z = 0; z < loop_count; z++) {
    cc::extent<2> ext(h, w);
    texture<float, 2> stex(ext, tmp[0], h * w * 4, 32);
    const texture_view<const float, 2> sview(stex);
    texture<float, 2> dtex(ext, 32);

    parallel_for_each(ext, [&, sview, h, w, ext](cc::index<2> idx) restrict(amp) {
      float sum = 0;
      int j = 0;
      const int karr[] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
      for(int dy = -1; dy <= 1; dy++) {
        for(int dx = -1; dx <= 1; dx++) {
          sum += karr[j] * sview.sample(coord(idx[0] + dx, idx[1] + dy, ext));
          j++;
        }
      }
      dtex.set(idx, sum / 16);
    });
    cg::copy(dtex, tmp2[0], h * w * 4);
  }
  print_time(label, sw.lap() / loop_count);
  paral_for_pp_(tmp2, dst, [&](const auto& pp) { return uchar(pp(0, 0)); });
}
#endif //#if defined(_MSC_VER)

void f15(Mtx1b& src, Mtx1b& dst, int loop_count)
{
  std::string label = "15|Cベースの記述 ";

  using namespace std;

  int h = src.rows;
  int w = src.cols;
  int sstep = src.stepT();
  int dstep = dst.stepT();
  const uchar* psrc = src[0];
  uchar* pdst = dst[0];

  StopWatch sw;

  for(int z = 0; z < loop_count; z++) {
    src.extrapolate();
#pragma omp parallel for
    for(int y = 0; y < h; y++) {
      const uchar* ps = psrc + y * sstep;
      uchar* pd = pdst + y * dstep;
      for(int x = 0; x < w; x++) {
        *pd = (ps[-sstep - 1] + 2 * ps[-sstep] + ps[-sstep + 1] + 2 * ps[-1] + 4 * ps[0] +
               2 * ps[1] + ps[sstep - 1] + 2 * ps[sstep] + ps[sstep + 1]) /
            16;
        pd++;
        ps++;
      }
    }
  }
  print_time(label, sw.lap() / loop_count, "bp");
  //	print(dst);
}

void f16(Mtx1b& src, Mtx1b& dst, int loop_count)
{
  std::string label = "16|Cベースの記述 ";
  using namespace std;

  int h = src.rows;
  int w = src.cols;
  int sstep = src.stepT();
  int dstep = dst.stepT();
  const uchar* psrc = src[0];
  uchar* pdst = dst[0];

  StopWatch sw;

  for(int z = 0; z < loop_count; z++) {
    src.extrapolate();
#pragma omp parallel for
    for(int y = 0; y < h; y++) {
      const uchar* ps = psrc + y * sstep;
      uchar* pd = pdst + y * dstep;
      for(int x = 0; x < w; x++) {
        pd[x] =
            (ps[x - sstep - 1] + 2 * ps[x - sstep] + ps[x - sstep + 1] + 2 * ps[x - 1] + 4 * ps[x] +
             2 * ps[x + 1] + ps[x + sstep - 1] + 2 * ps[x + sstep] + ps[x + sstep + 1]) /
            16;
      }
    }
  }
  print_time(label, sw.lap() / loop_count, "bp");
  //	print(dst);
}

void f17(Mtx1b& src, Mtx1b& dst, int loop_count)
{
  std::string label = "17|Cベースの記述 (auto vectorizationねらい)";
  using namespace std;

  int h = src.rows;
  int w = src.cols;
  int sstep = src.stepT();
  int dstep = dst.stepT();
  uchar* psrc = src[0];
  uchar* pdst = dst[0];

  StopWatch sw;

  for(int z = 0; z < loop_count; z++) {
    src.extrapolate();  //c++のコードだが、計測をフェアにするため
#pragma omp parallel for
    for(int y = 0; y < h; y++) {
      uchar* ps = psrc + y * sstep;
      uchar* pd = pdst + y * dstep;

      uchar* ps1 = ps - sstep - 1;
      uchar* ps2 = ps - sstep;
      uchar* ps3 = ps - sstep + 1;
      uchar* ps4 = ps - 1;
      uchar* ps5 = ps + 1;
      uchar* ps6 = ps + sstep - 1;
      uchar* ps7 = ps + sstep;
      uchar* ps8 = ps + sstep + 1;

      for(int x = 0; x < w; x++) {
        pd[x] = (ps1[x] + 2 * ps2[x] + ps3[x] + 2 * ps4[x] + 4 * ps[x] + 2 * ps5[x] + ps6[x] +
                 2 * ps7[x] + ps8[x]) /
            16;
      }
    }
  }
  print_time(label, sw.lap() / loop_count, "bp");
  //	print(dst);
}

void f18(Mtx1b& src, Mtx1b& dst, int loop_count)
{
  std::string label = "18|Cベースの記述 (auto vectorizationねらい)";
  using namespace std;

  int h = src.rows;
  int w = src.cols;
  int sstep = src.stepT();
  int dstep = dst.stepT();
  uchar* psrc = src[0];
  uchar* pdst = dst[0];

  static const int ker[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};

  StopWatch sw;

  for(int z = 0; z < loop_count; z++) {
    src.extrapolate();  //c++のコードだが、計測をフェアにするため
#pragma omp parallel for
    for(int y = 0; y < h; y++) {
      uchar* pd = pdst + y * dstep;
      uchar* ps[9];
      ps[4] = psrc + y * sstep;
      ps[0] = ps[4] - sstep - 1;
      ps[1] = ps[4] - sstep;
      ps[2] = ps[4] - sstep + 1;
      ps[3] = ps[4] - 1;
      ps[5] = ps[4] + 1;
      ps[6] = ps[4] + sstep - 1;
      ps[7] = ps[4] + sstep;
      ps[8] = ps[4] + sstep + 1;

      int sum[4096] = {0};
      for(int j = 0; j < 9; j++) {
        for(int x = 0; x < w; x++) {
          sum[x] += ker[j] * ps[j][x];
        }
      }
      for(int x = 0; x < w; x++) {
        pd[x] = sum[x] / 16;
      }
    }
  }
  print_time(label, sw.lap() / loop_count, "bp");
  //  print(dst);
}

void f0(Mtx1b& src, Mtx1b& dst, int loop_count)
{
  using namespace std;
  StopWatch sw;
  std::string label = "0|cv::GausianBlur ";
  for(int z = 0; z < loop_count; z++) {
    GaussianBlur(src, dst, cv::Size(3, 3), 0);
  }
  print_time(label, sw.lap() / loop_count, "");
}

int main()
{
  using namespace std;

  auto src = Mtx1b::createWithBorder(1080, 1920, 1);
  //auto src = Mtx1b::createWithBorder(5, 10, 1);
  auto ref = Mtx1b::createWithBorder(src.size(), 1);
  Mtx1b dst(src.size());
  fill_test_value(src);
  int h = src.rows;
  int w = src.cols;
  int loop_count = 1000;

  f0(src, ref, loop_count);
  f1(src, ref, loop_count / 10);
  ref.extrapolate();

  f2(src, dst, loop_count);
  check_value(dst, ref);
  f3(src, dst, loop_count);
  check_value(dst, ref);
  f4(src, dst, loop_count);
  check_value(dst, ref);
  f5(src, dst, loop_count);
  check_value(dst, ref);
  f6(src, dst, loop_count);
  check_value(dst, ref);
  f7(src, dst, loop_count);
  check_value(dst, ref);
  f7_1(src, dst, loop_count);
  check_value(dst, ref);
  f7_2(src, dst, loop_count);
  check_value(dst, ref);
  f7_3(src, dst, loop_count);
  check_value(dst, ref);
  f7_4(src, dst, loop_count);
  check_value(dst, ref);
//  f7_5(src, dst, loop_count);
//  check_value(dst, ref);
  f9(src, dst, loop_count);
  check_value(dst, ref);
  f10(src, dst, loop_count);
  check_value(dst, ref);
  f10_1(src, dst, loop_count);
  check_value(dst, ref);
  f10_2(src, dst, loop_count);
  check_value(dst, ref);
  f10_3(src, dst, loop_count);
  check_value(dst, ref);
  f10_4(src, dst, loop_count);
  check_value(dst, ref);
  f10_5(src, dst, loop_count);
  check_value(dst, ref);
  f11(src, dst, loop_count);
  check_value(dst, ref);
  f12(src, dst, loop_count);
  check_value(dst, ref);
  f13(src, dst, loop_count);
  check_value(dst, ref);
  f13_1(src, dst, loop_count);
  check_value(dst, ref);
  f13_2(src, dst, loop_count);
  check_value(dst, ref);
  f14(src, dst, loop_count);
  check_value(dst, ref);
  f14_1(src, dst, loop_count);
  check_value(dst, ref);
  f14_2(src, dst, loop_count);
  check_value(dst, ref);
  f15(src, dst, loop_count);
  check_value(dst, ref);
  f16(src, dst, loop_count);
  check_value(dst, ref);
  f17(src, dst, loop_count);
  check_value(dst, ref);
  f18(src, dst, loop_count);
  check_value(dst, ref);

  return 1;
}

int main5()
{
  using namespace std;

  //入力用データ生成
  Mtx1b mtx(10000, 10000);
  //fill_value(mtx);
  //print(mtx);
  auto mtx2 = Mtx1b::createWithBorder(mtx.size(), 1);
  fill_test_value(mtx2);
  mtx2.extrapolate();

  //cout << "入力" << endl;
  //print(mtx2);
  StopWatch sw;
  paral_for_pp_(mtx2, mtx, [](const auto& ps) {
    return std::clamp(ps(-1, -1) - ps(-1, 1) + ps(0, -1) * 2 - ps(0, 1) * 2 + ps(1, -1) - ps(1, 1),
                      0, 255);
  });
  cout << fixed << setprecision(2) << sw.lap() << endl;
  cv::Mat1b mat1(10000, 10000);
  fill_test_value(mat1);
  cv::Mat mat2;
  sw.lap();
  Sobel(mat1, mat2, CV_8U, 1, 0, 3);
  cout << fixed << setprecision(2) << sw.lap() << endl;
  // cout << endl;
  //cout << "結果" << endl;
  //print(mtx);

  return 0;
}

int main3()
{
  using namespace std;

  //入力用データ生成
  Mtx1w mtx(10, 10);
  fill_test_value(mtx);

  //(1)(1)Offset値参照によるアドレス計算の省略機能
  cout << "(1)Offset値参照によるアドレス計算の省略機能" << endl;
  Mtx1f mtxf(mtx.size());
  assert(mtxf.stepT() == mtx.stepT());
  for(int y = 0; y < mtx.rows; y++) {
    auto os = mtx.calcOffset(y, 0);
    for(int x = 0; x < mtx.cols; x++) {
      mtxf(os) = mtx(os);
      os++;
    }
  }
  cout << "mtx(1,1)=" << mtx(1, 1) << endl;
  cout << "mtxf(1,1)=" << mtxf(1, 1) << endl;
  cout << endl;

  //(2)画面外の画素値の参照
  cout << "(2)画面外の画素値の参照" << endl;
  auto mtx_ext = Mtx1i::createWithBorder(10, 10, 1, 2);
  fill_test_value(mtx_ext);
  mtx_ext.extrapolate();
  print(mtx_ext);

  cv::Mat_<int> mat = mtx_ext;
  cout << "mtx_ext(0,0)=" << mtx_ext(0, 0) << endl;
  cout << "mat(0,0)=" << mat(0, 0) << endl;
  cout << endl;
  //(3)実数座標値による補完画素値の取得

  cout << mtxf(11.2f, 11.2f);
  cout << mtxf(12.1f, 12.1f);

  cout << "(3)実数座標値による補完画素値の取得" << endl;
  cout << "mtxf(0,0)=" << mtxf(0, 0) << endl;
  cout << "mtxf(0,1)=" << mtxf(0, 1) << endl;
  cout << "mtxf(1,0)=" << mtxf(1, 0) << endl;
  cout << "mtxf(1,1)=" << mtxf(1, 1) << endl;
  cout << "mtxf(0.5,0.5)=" << mtxf(0.5f, 0.5f) << endl;
  cout << "mtxf(0.9,0.4)=" << mtxf(0.9f, 0.4f) << endl;
  cout << "mtx(0.9,0.4)=" << mtx(0.9f, 0.4f) << endl;
  return 0;
}

int main10()
{
  cv::Mat3b src = cv::imread("c:\\temp\\picture.jpg");
  cv::Mat3f dst(src.size());
  /*
		src.forEach([&dst](const auto& pix, const int* pos) {
			dst(pos[0],pos[1]) = conv<Vec3f>(pix, [](const auto& v) {
				return v * 0.5f;
			});
		});
	*/
  src.forEach([&dst](const auto& pix, const int* pos) {
    conv(pix, dst(pos[0], pos[1]), [](const auto& s, auto& d) { d = s * 0.5f; });
  });

  Mtx_<float> mf(10, 5);
  for(int y = 0; y < mf.rows; y++) {
    for(int x = 0; x < mf.cols; x++) {
      auto offset = mf.calcOffset(y, x);
      mf(offset) = 1.f;
    }
  }

  Mtx_<uchar> mb(10, 5);
  for(int y = 0; y < mf.rows; y++) {
    for(int x = 0; x < mf.cols; x++) {
      auto offset = mf.calcOffset(y, x);
      mf(offset) = mb(offset);
    }
  }

  //	imshow("dark image", dst);
  cv::waitKey();

  return 0;
}
