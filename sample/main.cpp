#include <iostream>
#include <execution>
#include <algorithm>
#include "cvx.h"

using namespace cvx;

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

int main()
{
  using namespace std;



  auto src = Mtx1b::createWithBorder(1920, 1080, 1);
  Mtx1b dst(src.size());
  Mtx1b ref(src.size());
  fill_value(src);
  int h = src.rows;
  int w = src.cols;
  int loop_count = 10;

  src.extrapolate();
  // print(src);

  Kernel<int, 1> ker;
  int arr[]{1, 2, 1, 2, 4, 2, 1, 2, 1};
  ker.copy(arr);

  // for_dydx(1, 1, [&](int dy, int dx) { cout << k(dy, dx) << endl;
  // });
  StopWatch sw;
  cout << "case0: ";
  cv::GaussianBlur(src, dst, cv::Size(3, 3), 0);
  cout << sw.lap() << endl << endl;

  cout << "case1: ";
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
  cout << sw.lap() << endl;
  //print(ref);
  cout << "case2: ";
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
  cout << sw.lap() << endl;
  cout << "check: " << check_value(dst, ref) << endl << endl;

  cout << "case3: ";
  for(int z = 0; z < loop_count; z++) {
    src.extrapolate();
    for(int y = 0; y < h; y++) {
      auto pp = src.getPixPtr(y, 0);
      for(int x = 0; x < w; x++) {
        dst(y, x) = (4 * pp(0, 0) + 2 * (pp(0, -1) + pp(-1, 0) + pp(0, 1) + pp(1, 0)) +
                     (pp(-1, -1) + pp(-1, 1) + pp(1, 1) + pp(1, -1))) /
            16;
        pp++;
      }
    }
  }
  cout << sw.lap() << endl;
  cout << "check: " << check_value(dst, ref) << endl << endl;

  cout << "case4: ";
  for(int z = 0; z < loop_count; z++) {
    src.extrapolate();
    for(int y = 0; y < h; y++) {
      auto pp = src.getPixPtr(y, 0);
      for(int x = 0; x < w; x++) {
        dst(y, x) = (pp(-1, -1) + 2 * pp(-1, 0) + pp(-1, 1) + 2 * pp(0, -1) + 4 * pp(0, 0) +
                     2 * pp(0, 1) + pp(1, -1) + 2 * pp(1, 0) + pp(1, 1)) /
            16;
        pp++;
      }
    }
  }
  cout << sw.lap() << endl;
  cout << "check: " << check_value(dst, ref) << endl << endl;

  cout << "case5: ";
  for(int z = 0; z < loop_count; z++) {
    src.extrapolate();
    paral_for_pp(src, dst, [](const auto& pp) {
      return (pp(-1, -1) + 2 * pp(-1, 0) + pp(-1, 1) + 2 * pp(0, -1) + 4 * pp(0, 0) + 2 * pp(0, 1) +
              pp(1, -1) + 2 * pp(1, 0) + pp(1, 1)) /
          16;
    });
  }
  cout << sw.lap() << endl;
  cout << "check: " << check_value(dst, ref) << endl << endl;

  cout << "case6: ";
  src.extrapolate();
  for(int z = 0; z < loop_count; z++) {
    paral_for_pp(src, dst, [&](const auto& pp) {
#if 0
      int sum = 0;
      for_dydx(ker.radius, ker.radius, [&](int dy, int dx) { sum += pp(dy, dx) * ker(dy, dx); });
      return sum / 16;
#elif 0
      int sum = 0;
      ker.for_([&](int kk, int dy, int dx) { sum += kk * pp(dy, dx); });
      return sum / 16;
#else
      return ker.convolute(pp) / 16;

#endif
    });
  }
  cout << sw.lap() << endl;
  //print(dst);
  cout << "check: " << check_value(dst, ref) << endl << endl;

  cout << "case7: ";
  src.extrapolate();
  for(int z = 0; z < loop_count; z++) {
#pragma omp parallel for
    for(int y = 0; y < h; y++) {
      auto pd = dst[y];
      int sum[4096] = {};
#if 1
      //    for_pd_idx<32>(w,  sum, [&](auto& psum, int d) { psum[d] = 0; });
      for_dydx(ker.radius, ker.radius, [&](int dy, int dx) {
        int k = ker(dy, dx);
        auto ps = src[0] + src.stepT() * (dy + y) + dx;
        for_psd_idx<32>(w, ps, sum, [&](auto& ps, auto& psum, int d) { psum[d] += k * ps[d]; });
      });
      for_psd_idx<32>(w, sum, pd, [&](auto& psum, auto& pd, int d) { pd[d] = psum[d] / 16; });
    }
#elif 1
      //      for_psd_idx<32>(w, sum, pd, [&](auto& ps, auto& pd, int d) { pd[d] = 0; });
      ker.for_([&](auto&& k, int dy, int dx) {
        auto ps = src[0] + src.stepT() * (dy + y) + dx;
        for_psd_idx<32>(w, ps, sum, [&](auto& ps, auto& pd, int d) { pd[d] += k * ps[d]; });
      });
      for_psd_idx<32>(w, sum, pd, [&](auto& ps, auto& pd, int d) { pd[d] = ps[d] / 16; });
#else

#endif
  }
  cout << sw.lap() << endl;
  cout << "check: " << check_value(dst, ref) << endl << endl;

  cout << "case8: ";
  for(int z = 0; z < loop_count; z++) {
    src.extrapolate();
#pragma omp parallel for
    for(int y = 0; y < h; y++) {
      auto pd = dst[y];
      int sum[4096] = {};
      for_dydx(ker.radius, ker.radius, [&](int dy, int dx) {
        int k = ker(dy, dx);
        auto ps = src[0] + src.stepT() * (dy + y) + dx;
        for_psd_idx<32>(w, ps, sum, [&](auto& ps, auto& psum, int d) { psum[d] += k * ps[d]; });
      });
      for_psd_idx<32>(w, sum, pd, [&](auto& psum, auto& pd, int d) { pd[d] = psum[d] / 16; });
    }
  }
  cout << sw.lap() << endl;
  cout << "check: " << check_value(dst, ref) << endl << endl;

  cout << "case9: ";
  SepKernel<int, 1> sker(2, 1);
  for(int z = 0; z < loop_count; z++) {
    src.extrapolate();
    auto tmp = Mtx1i::createWithBorder(src.size(), sker.radius);
    paral_for_pp(src, tmp, [&](const auto& pp) { return sker.convolute(pp, false); });
    tmp.extrapolate();
    paral_for_pp(tmp, dst, [&](const auto& pp) { return sker.convolute(pp, true) / 16; });
  }
  cout << sw.lap() << endl;
  cout << "check: " << check_value(dst, ref) << endl << endl;
}

int main5()
{
  using namespace std;

  //入力用データ生成
  Mtx1b mtx(10000, 10000);
  //fill_value(mtx);
  //print(mtx);
  auto mtx2 = Mtx1b::createWithBorder(mtx.size(), 1);
  fill_value(mtx2);
  mtx2.extrapolate();

  //cout << "入力" << endl;
  //print(mtx2);
  StopWatch sw;
  paral_for_pp(mtx2, mtx, [](const auto& ps) {
    return std::clamp(ps(-1, -1) - ps(-1, 1) + ps(0, -1) * 2 - ps(0, 1) * 2 + ps(1, -1) - ps(1, 1),
                      0, 255);
  });
  cout << sw.lap() << endl;
  cv::Mat1b mat1(10000, 10000);
  fill_value(mat1);
  cv::Mat mat2;
  sw.lap();
  Sobel(mat1, mat2, CV_8U, 1, 0, 3);
  cout << sw.lap() << endl;
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
  fill_value(mtx);

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
  fill_value(mtx_ext);
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
