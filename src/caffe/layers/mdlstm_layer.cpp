#include <vector>
#include <iostream>
#include <iomanip>
#include <numeric>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#ifdef _OPENMP
#include "omp.h"
#endif

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(const Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
void MDLSTMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
const int num_hidden = this->layer_param_.mdlstm_param().num_hidden();
const int num_threads = this->layer_param_.mdlstm_param().num_threads();
//bias_term_ = this->layer_param_.mdlstm_param().bias_term();
num_ = num_hidden;
threads_ = num_threads;
//K_ = bottom[0]->count() / bottom[0]->num();
// Check if we need to set up the weights
if (this->blobs_.size() > 0) {
  LOG(INFO) << "Skipping parameter initialization";
} else {
  this->blobs_.resize(10);
  // 0: Whzx   Cell output to cell input along x
  // 1: Whzy   Cell output to cell input along y
  // 2: Whix   Cell output to input gate along x
  // 3: Whiy   Cell output to input gate along y
  // 4: Whf1x  Cell output to forget gate 1 along x
  // 5: Whf1y  Cell output to forget gate 1 along y
  // 6: Whf2x  Cell output to forget gate 2 along x
  // 7: Whf2y  Cell output to forget gate 2 along y
  // 8: Whox   Cell output to output gate along x
  // 9: Whoy   Cell output to output gate along y
  // Initialize the weights
  for (int i=0; i<10; ++i) {
    this->blobs_[i].reset(new Blob<Dtype>(1, 1, num_, num_));
  }

  // fill the weights
  shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(this->layer_param_.mdlstm_param().weight_filler()));
  for (int i=0; i<10; i++) {
    weight_filler->Fill(this->blobs_[i].get());
  }

}  // parameter initialization
this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void MDLSTMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
// Figure out the dimensions
//M_ = bottom[0]->num();
CHECK_EQ(bottom.size(), top.size()) << "Bottom size "
"must be the same as top size.";
bh_.resize(bottom.size());
bi_.resize(bottom.size());
bz_.resize(bottom.size());
bo_.resize(bottom.size());
bf1_.resize(bottom.size());
bf2_.resize(bottom.size());
s_.resize(bottom.size());
for (int bottom_id=0; bottom_id<bottom.size(); ++bottom_id) {
  CHECK_EQ(bottom[bottom_id]->channels(), 5 * num_) << "Input size "
  "must be 5 times hidden units.";
  top[bottom_id]->Reshape(bottom[bottom_id]->num(), num_, bottom[bottom_id]->height(), bottom[bottom_id]->width());
}

}

template <typename Dtype>
void MDLSTMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
for (int bottom_id = 0; bottom_id < bottom.size(); ++bottom_id) {
  bh_[bottom_id].reset(new Blob<Dtype>(bottom[bottom_id]->num(), num_, bottom[bottom_id]->height(),
        bottom[bottom_id]->width()));
  bi_[bottom_id].reset(new Blob<Dtype>(bottom[bottom_id]->num(), num_, bottom[bottom_id]->height(),
        bottom[bottom_id]->width()));
  bz_[bottom_id].reset(new Blob<Dtype>(bottom[bottom_id]->num(), num_, bottom[bottom_id]->height(),
        bottom[bottom_id]->width()));
  bo_[bottom_id].reset(new Blob<Dtype>(bottom[bottom_id]->num(), num_, bottom[bottom_id]->height(),
        bottom[bottom_id]->width()));
  bf1_[bottom_id].reset(new Blob<Dtype>(bottom[bottom_id]->num(), num_, bottom[bottom_id]->height(),
        bottom[bottom_id]->width()));
  bf2_[bottom_id].reset(new Blob<Dtype>(bottom[bottom_id]->num(), num_, bottom[bottom_id]->height(),
        bottom[bottom_id]->width()));
  s_[bottom_id].reset(new Blob<Dtype>(bottom[bottom_id]->num(), num_, bottom[bottom_id]->height(),
        bottom[bottom_id]->width()));
  const Dtype *bottom_data = bottom[bottom_id]->cpu_data();
  Dtype *top_data = top[bottom_id]->mutable_cpu_data();
  const Dtype *Whzx = this->blobs_[0]->cpu_data();
  const Dtype *Whzy = this->blobs_[1]->cpu_data();
  const Dtype *Whix = this->blobs_[2]->cpu_data();
  const Dtype *Whiy = this->blobs_[3]->cpu_data();
  const Dtype *Whf1x = this->blobs_[4]->cpu_data();
  const Dtype *Whf1y = this->blobs_[5]->cpu_data();
  const Dtype *Whf2x = this->blobs_[6]->cpu_data();
  const Dtype *Whf2y = this->blobs_[7]->cpu_data();
  const Dtype *Whox = this->blobs_[8]->cpu_data();
  const Dtype *Whoy = this->blobs_[9]->cpu_data();
  Dtype *bh = bh_[bottom_id]->mutable_cpu_data();
  Dtype *bi = bi_[bottom_id]->mutable_cpu_data();
  Dtype *bz = bz_[bottom_id]->mutable_cpu_data();
  Dtype *bo = bo_[bottom_id]->mutable_cpu_data();
  Dtype *bf1 = bf1_[bottom_id]->mutable_cpu_data();
  Dtype *bf2 = bf2_[bottom_id]->mutable_cpu_data();
  Dtype *s = s_[bottom_id]->mutable_cpu_data();

  int start_x, end_x, inc_x;
  int start_y, end_y, inc_y;
  switch (this->layer_param_.mdlstm_param().vertical_dir()) {
    case MDLSTMParameter_VerticalDirection_DOWN:
      start_x = 0;
      end_x = bottom[bottom_id]->height() - 1;
      inc_x = 1;
      break;
    case MDLSTMParameter_VerticalDirection_UP:
      start_x = bottom[bottom_id]->height() - 1;
      end_x = 0;
      inc_x = -1;
      break;
    default:
      LOG(FATAL) << "Unknown vertical direction.";
  }
  switch (this->layer_param_.mdlstm_param().horizontal_dir()) {
    case MDLSTMParameter_HorizontalDirection_RIGHT:
      start_y = 0;
      end_y = bottom[bottom_id]->width() - 1;
      inc_y = 1;
      break;
    case MDLSTMParameter_HorizontalDirection_LEFT:
      start_y = bottom[bottom_id]->width() - 1;
      end_y = 0;
      inc_y = -1;
      break;
    default:
      LOG(FATAL) << "Unknown horizontal direction.";
  }
#ifdef _OPENMP
    omp_set_num_threads(threads_);
    #pragma omp parallel for
#endif
  for (int i = 0; i < bottom[bottom_id]->num(); ++i) {
    for (int x = start_x; (start_x == 0) ? x <= end_x : x >= end_x; x += inc_x) {
      for (int y = start_y; (start_y == 0) ? y <= end_y : y >= end_y; y += inc_y) {
        Dtype alpha_x = x != start_x ? (Dtype)inc_x : 0.;
        Dtype alpha_y = y != start_y ? (Dtype)inc_y : 0.;
        //using parameter beta=1 to accumulate the result of matrix multiplications
        //using the alpha parameter to control the boundaries
        //az[x,y,:] = input_data_0[x,y,:] + np.dot(in_x, Whzx) + np.dot(in_y, Whzy)
        const int ld_step = bottom[bottom_id]->width() * bottom[bottom_id]->height();
        strided_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, 1, num_, abs(alpha_x), Whzx, num_,
                bh + bh_[bottom_id]->offset(i, 0, x - alpha_x, y), ld_step, (Dtype) 1.,
                bz + bz_[bottom_id]->offset(i, 0, x, y), ld_step);
        strided_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, 1, num_, abs(alpha_y), Whzy, num_,
                bh + bh_[bottom_id]->offset(i, 0, x, y - alpha_y), ld_step, (Dtype) 1.,
                bz + bz_[bottom_id]->offset(i, 0, x, y), ld_step);
        strided_cpu_axpby<Dtype>(num_, (Dtype) 1., bottom_data + bottom[bottom_id]->offset(i, 0, x, y),
                ld_step, (Dtype) 1., bz + bz_[bottom_id]->offset(i, 0, x, y), ld_step);

        int start = bz_[bottom_id]->offset(i, 0, x, y);
        int end = bz_[bottom_id]->offset(i, num_, x, y);
        for (int e = start; e < end; e += ld_step) {
            bz[e] = tanh(bz[e]);
        }


        //ai[x,y,:] = input_data_1[x,y,:] + np.dot(in_x, Whix) + np.dot(in_y, Whiy)
        strided_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, 1, num_, abs(alpha_x), Whix, num_,
                bh + bh_[bottom_id]->offset(i, 0, x - alpha_x, y), ld_step, (Dtype) 1.,
                bi + bi_[bottom_id]->offset(i, 0, x, y), ld_step);
        strided_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, 1, num_, abs(alpha_y), Whiy, num_,
                bh + bh_[bottom_id]->offset(i, 0, x, y - alpha_y), ld_step, (Dtype) 1.,
                bi + bi_[bottom_id]->offset(i, 0, x, y), ld_step);
        strided_cpu_axpby<Dtype>(num_, (Dtype) 1., bottom_data + bottom[bottom_id]->offset(i, num_, x, y),
                ld_step, (Dtype) 1., bi + bi_[bottom_id]->offset(i, 0, x, y), ld_step);

        start = bi_[bottom_id]->offset(i, 0, x, y);
        end = bi_[bottom_id]->offset(i, num_, x, y);
        for (int e = start; e < end; e += ld_step) {
            bi[e] = sigmoid(bi[e]);
        }


        //ao[x,y,:] = input_data_2[x,y,:] + np.dot(in_x, Whox) + np.dot(in_y, Whoy)
        strided_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, 1, num_, abs(alpha_x), Whox, num_,
                bh + bh_[bottom_id]->offset(i, 0, x - alpha_x, y), ld_step, (Dtype) 1.,
                bo + bo_[bottom_id]->offset(i, 0, x, y), ld_step);
        strided_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, 1, num_, abs(alpha_y), Whoy, num_,
                bh + bh_[bottom_id]->offset(i, 0, x, y - alpha_y), ld_step, (Dtype) 1.,
                bo + bo_[bottom_id]->offset(i, 0, x, y), ld_step);
        strided_cpu_axpby<Dtype>(num_, (Dtype) 1., bottom_data + bottom[bottom_id]->offset(i, 2 * num_, x, y),
                ld_step, (Dtype) 1., bo + bo_[bottom_id]->offset(i, 0, x, y), ld_step);

        start = bo_[bottom_id]->offset(i, 0, x, y);
        end = bo_[bottom_id]->offset(i, num_, x, y);
        for (int e = start; e < end; e += ld_step) {
            bo[e] = sigmoid(bo[e]);
        }


        //af1[x,y,:] = input_data_3[x,y,:] + np.dot(in_x, Whf1x) + np.dot(in_y, Whf1y)
        strided_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, 1, num_, abs(alpha_x), Whf1x, num_,
                bh + bh_[bottom_id]->offset(i, 0, x - alpha_x, y), ld_step, (Dtype) 1.,
                bf1 + bf1_[bottom_id]->offset(i, 0, x, y), ld_step);
        strided_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, 1, num_, abs(alpha_y), Whf1y, num_,
                bh + bh_[bottom_id]->offset(i, 0, x, y - alpha_y), ld_step, (Dtype) 1.,
                bf1 + bf1_[bottom_id]->offset(i, 0, x, y), ld_step);
        strided_cpu_axpby<Dtype>(num_, (Dtype) 1., bottom_data + bottom[bottom_id]->offset(i, 3 * num_, x, y),
                ld_step, (Dtype) 1., bf1 + bf1_[bottom_id]->offset(i, 0, x, y), ld_step);

        start = bf1_[bottom_id]->offset(i, 0, x, y);
        end = bf1_[bottom_id]->offset(i, num_, x, y);
        for (int e = start; e < end; e += ld_step) {
            bf1[e] = sigmoid(bf1[e]);
        }

        //af2[x,y,:] = input_data_4[x,y,:] + np.dot(in_x, Whf2x) + np.dot(in_y, Whf2y)
        strided_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, 1, num_, abs(alpha_x), Whf2x, num_,
                bh + bh_[bottom_id]->offset(i, 0, x - alpha_x, y), ld_step, (Dtype) 1.,
                bf2 + bf2_[bottom_id]->offset(i, 0, x, y), ld_step);
        strided_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, 1, num_, abs(alpha_y), Whf2y, num_,
                bh + bh_[bottom_id]->offset(i, 0, x, y - alpha_y), ld_step, (Dtype) 1.,
                bf2 + bf2_[bottom_id]->offset(i, 0, x, y), ld_step);
        strided_cpu_axpby<Dtype>(num_, (Dtype) 1., bottom_data + bottom[bottom_id]->offset(i, 4 * num_, x, y),
                ld_step, (Dtype) 1., bf2 + bf2_[bottom_id]->offset(i, 0, x, y), ld_step);

        start = bf2_[bottom_id]->offset(i, 0, x, y);
        end = bf2_[bottom_id]->offset(i, num_, x, y);
        for (int e = start; e < end; e += ld_step) {
            bf2[e] = sigmoid(bf2[e]);
        }


        //s[x,y,:] = bi[x,y,:] * bz[x,y,:] + s_x * bf1[x,y,:] + s_y * bf2[x,y,:]
        start = s_[bottom_id]->offset(i, 0, x, y);
        end = s_[bottom_id]->offset(i, num_, x, y);
        int idx_sx = s_[bottom_id]->offset(i, 0, x - alpha_x, y);
        int idx_sy = s_[bottom_id]->offset(i, 0, x, y - alpha_y);
        for (int e = start; e < end; e += ld_step) {
          s[e] = bi[e] * bz[e] +
                  (x != start_x ? s[idx_sx] : 0.) * bf1[e] +
                  (y != start_y ? s[idx_sy] : 0.) * bf2[e];
            if (isnan(s[e]) || isinf(s[e])) {
                LOG(INFO) << "x = " << x <<
                            " y = " << y << " s[" << e << "] = " << s[e] << " s[" << idx_sx << "] = " << s[idx_sx] << " s[" << idx_sy << "] = " << s[idx_sy];
                LOG(INFO) << x << " " << y << " " << i << bi[e] << " " << bz[e] << " bf1 = " << bf1[e] << " bf2 = " << bf2[e];
                char c = getchar();
            }
            // Clip state
//            if (s[e] > 10.)
//                s[e] = 10.;
//            else if (s[e] < -10.)
//                s[e] = -10.;
          idx_sx += ld_step;
          idx_sy += ld_step;
        }

        //bh[x,y,:] = bo[x,y,:] * h(s[x,y,:])
        start = bh_[bottom_id]->offset(i, 0, x, y);
        end = bh_[bottom_id]->offset(i, num_, x, y);
        for (int e = start; e < end; e += ld_step) {
          top_data[e] = bo[e] * tanh(s[e]);
          bh[e] = bo[e] * tanh(s[e]);
//          if (isnan(bh[e]) || isinf(bh[e])) {
//              LOG(INFO) << "bh block: " << x << " " << y << " " << i << tanh(s[e]) << " " << bo[e];
//              char c = getchar();
//          }
        }
      }
    }
  }
}
//Logging code for debug (to remove later)
//    std::ofstream log_s;
//    log_s.open("/Users/snake91/Desktop/s.txt", ios::app);
//    const int count = s_[0]->count();
//    const Dtype *data_s = s_[0]->cpu_data();
//    Dtype min = *std::min_element(data_s, data_s + count);
//    Dtype max = *std::max_element(data_s, data_s + count);
//    Dtype mean = std::accumulate(data_s, data_s + count, 0.) / count;
//    log_s << min << "," << max << "," << mean << std::endl;
//    for (int i = 0; i < count; ++i) {
//        log_s << std::setw(15) << data_s[i];
//        if ((i+1) % 160 == 0)
//            log_s << "\n";
//    }
//    log_s.close();
//    char c = getchar();
/*std::ofstream log_bi;
std::ofstream log_bz;
std::ofstream log_bo;
std::ofstream log_s;
std::ofstream log_bh;
std::ofstream log_bottom;
log_bi.open("/home/snake91/Desktop/bi.txt");
log_bz.open("/home/snake91/Desktop/bz.txt");
log_bo.open("/home/snake91/Desktop/bo.txt");
log_s.open("/home/snake91/Desktop/s.txt");
log_bh.open("/home/snake91/Desktop/bh.txt");
log_bottom.open("/home/snake91/Desktop/bottom.txt");
const int count = bi_[0]->count();

const Dtype *data_bi = bi_[0]->cpu_data();
const Dtype *data_bz = bz_[0]->cpu_data();
const Dtype *data_bo = bo_[0]->cpu_data();
const Dtype *data_s = s_[0]->cpu_data();
const Dtype *data_bh = bh_[0]->cpu_data();
const Dtype *data_bottom = bottom[0]->cpu_data();
for (int i = 0; i < count; ++i) {
    log_bi << data_bi[i] << std::endl;
    log_bz << data_bz[i] << std::endl;
    log_bo << data_bo[i] << std::endl;
    log_s << data_s[i] << std::endl;
    log_bh << data_bh[i] << std::endl;
    log_bottom << data_bottom[i] << std::endl;
}
log_bi.close();
log_bz.close();
log_bo.close();
log_s.close();
log_bh.close();
log_bottom.close();*/
}

template <typename Dtype>
void MDLSTMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int bottom_id = 0; bottom_id < bottom.size(); ++bottom_id) {

    const Dtype *bz_diff = bz_[bottom_id]->cpu_diff();
    const Dtype *bi_diff = bi_[bottom_id]->cpu_diff();
    const Dtype *bf1_diff = bf1_[bottom_id]->cpu_diff();
    const Dtype *bf2_diff = bf2_[bottom_id]->cpu_diff();
    const Dtype *bo_diff = bo_[bottom_id]->cpu_diff();
    Dtype *s_diff = s_[bottom_id]->mutable_cpu_diff();
    Dtype *bh_diff = bh_[bottom_id]->mutable_cpu_diff();
    const Dtype *Whzx = this->blobs_[0]->cpu_data();
    const Dtype *Whzy = this->blobs_[1]->cpu_data();
    const Dtype *Whix = this->blobs_[2]->cpu_data();
    const Dtype *Whiy = this->blobs_[3]->cpu_data();
    const Dtype *Whf1x = this->blobs_[4]->cpu_data();
    const Dtype *Whf1y = this->blobs_[5]->cpu_data();
    const Dtype *Whf2x = this->blobs_[6]->cpu_data();
    const Dtype *Whf2y = this->blobs_[7]->cpu_data();
    const Dtype *Whox = this->blobs_[8]->cpu_data();
    const Dtype *Whoy = this->blobs_[9]->cpu_data();
    Dtype *Whzx_diff = this->blobs_[0]->mutable_cpu_diff();
    Dtype *Whzy_diff = this->blobs_[1]->mutable_cpu_diff();
    Dtype *Whix_diff = this->blobs_[2]->mutable_cpu_diff();
    Dtype *Whiy_diff = this->blobs_[3]->mutable_cpu_diff();
    Dtype *Whf1x_diff = this->blobs_[4]->mutable_cpu_diff();
    Dtype *Whf1y_diff = this->blobs_[5]->mutable_cpu_diff();
    Dtype *Whf2x_diff = this->blobs_[6]->mutable_cpu_diff();
    Dtype *Whf2y_diff = this->blobs_[7]->mutable_cpu_diff();
    Dtype *Whox_diff = this->blobs_[8]->mutable_cpu_diff();
    Dtype *Whoy_diff = this->blobs_[9]->mutable_cpu_diff();
    const Dtype* top_diff = top[bottom_id]->cpu_diff();
    Dtype* bottom_diff = bottom[bottom_id]->mutable_cpu_diff();

    int start_x, end_x, inc_x;
    int start_y, end_y, inc_y;
    switch (this->layer_param_.mdlstm_param().vertical_dir()) {
      case MDLSTMParameter_VerticalDirection_DOWN:
        start_x = bottom[bottom_id]->height() - 1;
        end_x = 0;
        inc_x = -1;
        break;
      case MDLSTMParameter_VerticalDirection_UP:
        start_x = 0;
        end_x = bottom[bottom_id]->height() - 1;
        inc_x = 1;
        break;
      default:
        LOG(FATAL) << "Unknown vertical direction.";
    }
    switch (this->layer_param_.mdlstm_param().horizontal_dir()) {
      case MDLSTMParameter_HorizontalDirection_RIGHT:
        start_y = bottom[bottom_id]->width() - 1;
        end_y = 0;
        inc_y = -1;
        break;
      case MDLSTMParameter_HorizontalDirection_LEFT:
        start_y = 0;
        end_y = bottom[bottom_id]->width() - 1;
        inc_y = 1;
        break;
      default:
        LOG(FATAL) << "Unknown horizontal direction.";
    }

    const int ld_step = bottom[bottom_id]->width() * bottom[bottom_id]->height();
#ifdef _OPENMP
    omp_set_num_threads(threads_);
    #pragma omp parallel for
#endif
    for (int i = 0; i < bottom[bottom_id]->num(); ++i) {
      for (int x = start_x; start_x == 0 ? x <= end_x : x >= end_x; x += inc_x) {
        for (int y = start_y; start_y == 0 ? y <= end_y : y >= end_y; y += inc_y) {
          Dtype alpha_x = x != start_x ? -inc_x : 0.;
          Dtype alpha_y = y != start_y ? -inc_y : 0.;
//        bh[x,y,:] = delta[x,y,:] + (
//                    np.dot(delta_xo, Whox.T) + np.dot(delta_xi, Whix.T) + np.dot(delta_xf1, Whf1x.T) +
//                    np.dot(delta_xf2, Whf2x.T) + np.dot(delta_xz, Whzx.T) + np.dot(delta_yo, Whoy.T) +
//                    np.dot(delta_yi, Whiy.T) + np.dot(delta_yf1, Whf1y.T) +
//                    np.dot(delta_yf2, Whf2y.T) + np.dot(delta_yz, Whzy.T))
          strided_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_, 1, num_, abs(alpha_x), Whzx, num_,
                  bz_diff + bz_[bottom_id]->offset(i, 0, x + alpha_x, y),
                  ld_step, (Dtype) 1.,
                  bh_diff + bh_[bottom_id]->offset(i, 0, x, y), ld_step);
          strided_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_, 1, num_, abs(alpha_y), Whzy, num_,
                  bz_diff + bz_[bottom_id]->offset(i, 0, x, y + alpha_y),
                  ld_step, (Dtype) 1.,
                  bh_diff + bh_[bottom_id]->offset(i, 0, x, y), ld_step);
          strided_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_, 1, num_, abs(alpha_x), Whix, num_,
                  bi_diff + bi_[bottom_id]->offset(i, 0, x + alpha_x, y),
                  ld_step, (Dtype) 1.,
                  bh_diff + bh_[bottom_id]->offset(i, 0, x, y), ld_step);
          strided_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_, 1, num_, abs(alpha_y), Whiy, num_,
                  bi_diff + bi_[bottom_id]->offset(i, 0, x, y + alpha_y),
                  ld_step, (Dtype) 1.,
                  bh_diff + bh_[bottom_id]->offset(i, 0, x, y), ld_step);
          strided_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_, 1, num_, abs(alpha_x), Whox, num_,
                  bo_diff + bo_[bottom_id]->offset(i, 0, x + alpha_x, y),
                  ld_step, (Dtype) 1.,
                  bh_diff + bh_[bottom_id]->offset(i, 0, x, y), ld_step);
          strided_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_, 1, num_, abs(alpha_y), Whoy, num_,
                  bo_diff + bo_[bottom_id]->offset(i, 0, x, y + alpha_y),
                  ld_step, (Dtype) 1.,
                  bh_diff + bh_[bottom_id]->offset(i, 0, x, y), ld_step);
          strided_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_, 1, num_, abs(alpha_x), Whf1x, num_,
                  bf1_diff + bf1_[bottom_id]->offset(i, 0, x + alpha_x, y),
                  ld_step, (Dtype) 1.,
                  bh_diff + bh_[bottom_id]->offset(i, 0, x, y), ld_step);
          strided_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_, 1, num_, abs(alpha_y), Whf1y, num_,
                  bf1_diff + bf1_[bottom_id]->offset(i, 0, x, y + alpha_y),
                  ld_step, (Dtype) 1.,
                  bh_diff + bh_[bottom_id]->offset(i, 0, x, y), ld_step);
          strided_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_, 1, num_, abs(alpha_x), Whf2x, num_,
                  bf2_diff + bf2_[bottom_id]->offset(i, 0, x + alpha_x, y),
                  ld_step, (Dtype) 1.,
                  bh_diff + bh_[bottom_id]->offset(i, 0, x, y), ld_step);
          strided_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_, 1, num_, abs(alpha_y), Whf2y, num_,
                  bf2_diff + bf2_[bottom_id]->offset(i, 0, x, y + alpha_y),
                  ld_step, (Dtype) 1.,
                  bh_diff + bh_[bottom_id]->offset(i, 0, x, y), ld_step);
          strided_cpu_axpby<Dtype>(num_, (Dtype) 1., top_diff + top[bottom_id]->offset(i, 0, x, y),
                  ld_step, (Dtype) 1.,
                  bh_diff + bh_[bottom_id]->offset(i, 0, x, y), ld_step);


//        bo[x,y,:] = bh[x,y,:] * h(s[x,y,:]) * sigma_prime(ao[x,y,:])
          int start = bo_[bottom_id]->offset(i, 0, x, y);
          int end = bo_[bottom_id]->offset(i, num_, x, y);
          for (int e = start; e < end; e += ld_step) {
            bo_[bottom_id]->mutable_cpu_diff()[e] = bh_[bottom_id]->cpu_diff()[e] *
                    tanh(s_[bottom_id]->cpu_data()[e]) *
                    bo_[bottom_id]->cpu_data()[e] * (1 - bo_[bottom_id]->cpu_data()[e]);
          }

//        bs[x,y,:] = bh[x,y,:] * sigma(ao[x,y,:]) * h_prime(s[x,y,:]) + delta_xs*sigma(test_xf1) +
//                    delta_ys*sigma(test_yf2)
          start = s_[bottom_id]->offset(i, 0, x, y);
          end = s_[bottom_id]->offset(i, num_, x, y);
          int idx_sx = s_[bottom_id]->offset(i, 0, x + alpha_x, y);
          int idx_sy = s_[bottom_id]->offset(i, 0, x, y + alpha_y);
          for (int e = start; e < end; e += ld_step) {
            s_diff[e] = bh_[bottom_id]->cpu_diff()[e] *
                    bo_[bottom_id]->cpu_data()[e] *
                    (1 - pow(tanh(s_[bottom_id]->cpu_data()[e]), 2)) +
//                    (x < bottom[bottom_id]->height() - 1 ? s_[bottom_id]->cpu_diff()[idx_sx] : 0.) *
//                    (x < bottom[bottom_id]->height() - 1 ? bf1_[bottom_id]->cpu_data()[idx_sx] : sigmoid(0.)) +
//                    (y < bottom[bottom_id]->width() - 1 ? s_[bottom_id]->cpu_diff()[idx_sy] : 0.) *
//                    (y < bottom[bottom_id]->width() - 1 ? bf2_[bottom_id]->cpu_data()[idx_sy] : sigmoid(0.));
                    (x != start_x ? s_[bottom_id]->cpu_diff()[idx_sx] : 0.) *
                    (x != start_x ? bf1_[bottom_id]->cpu_data()[idx_sx] : sigmoid(0.)) +
                    (y != start_y ? s_[bottom_id]->cpu_diff()[idx_sy] : 0.) *
                    (y != start_y ? bf2_[bottom_id]->cpu_data()[idx_sy] : sigmoid(0.));
            idx_sx += ld_step;
            idx_sy += ld_step;
          }

//        bf1[x,y,:] = sigma_prime(af1[x,y,:]) * bs[x,y,:] * scx
//        bf2[x,y,:] = sigma_prime(af2[x,y,:]) * bs[x,y,:] * scy
          start = bo_[bottom_id]->offset(i, 0, x, y);
          end = bo_[bottom_id]->offset(i, num_, x, y);
//          idx_sx = s_[bottom_id]->offset(i, 0, x - 1, y);
//          idx_sy = s_[bottom_id]->offset(i, 0, x, y - 1);
          idx_sx = s_[bottom_id]->offset(i, 0, x + inc_x, y);
          idx_sy = s_[bottom_id]->offset(i, 0, x, y + inc_y);
          for (int e = start; e < end; e += ld_step) {
            bf1_[bottom_id]->mutable_cpu_diff()[e] = s_[bottom_id]->cpu_diff()[e] *
                    (x != end_x ? s_[bottom_id]->cpu_data()[idx_sx] : 0.) *
                    bf1_[bottom_id]->cpu_data()[e] * (1 - bf1_[bottom_id]->cpu_data()[e]);
            bf2_[bottom_id]->mutable_cpu_diff()[e] = s_[bottom_id]->cpu_diff()[e] *
                    (y != end_y ? s_[bottom_id]->cpu_data()[idx_sy] : 0.) *
                    bf2_[bottom_id]->cpu_data()[e] * (1 - bf2_[bottom_id]->cpu_data()[e]);
            idx_sx += ld_step;
            idx_sy += ld_step;
          }

//        bi[x,y,:] = sigma_prime(ai[x,y,:]) * g(az[x,y,:]) * bs[x,y,:]
          start = bi_[bottom_id]->offset(i, 0, x, y);
          end = bi_[bottom_id]->offset(i, num_, x, y);
          for (int e = start; e < end; e += ld_step) {
              bi_[bottom_id]->mutable_cpu_diff()[e] = s_[bottom_id]->cpu_diff()[e] *
                      bz_[bottom_id]->cpu_data()[e] *
                      bi_[bottom_id]->cpu_data()[e] * (1 - bi_[bottom_id]->cpu_data()[e]);
          }

//        bz[x,y,:] = g_prime(az[x,y,:]) * sigma(ai[x,y,:]) * bs[x,y,:]
          start = bz_[bottom_id]->offset(i, 0, x, y);
          end = bz_[bottom_id]->offset(i, num_, x, y);
          for (int e = start; e < end; e += ld_step) {
              bz_[bottom_id]->mutable_cpu_diff()[e] = s_[bottom_id]->cpu_diff()[e] *
                      bi_[bottom_id]->cpu_data()[e] *
                      (1 - pow(bz_[bottom_id]->cpu_data()[e], 2));
          }

        }
      }
      const int size = bottom[bottom_id]->height() * bottom[bottom_id]->width() * num_;
      caffe_copy(size, bz_diff + bz_[bottom_id]->offset(i, 0, 0, 0),
              bottom_diff + bottom[bottom_id]->offset(i, 0, 0, 0));
      caffe_copy(size, bi_diff + bi_[bottom_id]->offset(i, 0, 0, 0),
              bottom_diff + bottom[bottom_id]->offset(i, num_, 0, 0));
      caffe_copy(size, bo_diff + bo_[bottom_id]->offset(i, 0, 0, 0),
              bottom_diff + bottom[bottom_id]->offset(i, 2 * num_, 0, 0));
      caffe_copy(size, bf1_diff + bf1_[bottom_id]->offset(i, 0, 0, 0),
              bottom_diff + bottom[bottom_id]->offset(i, 3 * num_, 0, 0));
      caffe_copy(size, bf2_diff + bf2_[bottom_id]->offset(i, 0, 0, 0),
              bottom_diff + bottom[bottom_id]->offset(i, 4 * num_, 0, 0));
    }


    // Weights update
    memset(Whzx_diff, 0, sizeof(Dtype) * num_ * num_);
    memset(Whzy_diff, 0, sizeof(Dtype) * num_ * num_);
    memset(Whix_diff, 0, sizeof(Dtype) * num_ * num_);
    memset(Whiy_diff, 0, sizeof(Dtype) * num_ * num_);
    memset(Whf1x_diff, 0, sizeof(Dtype) * num_ * num_);
    memset(Whf1y_diff, 0, sizeof(Dtype) * num_ * num_);
    memset(Whf2x_diff, 0, sizeof(Dtype) * num_ * num_);
    memset(Whf2y_diff, 0, sizeof(Dtype) * num_ * num_);
    memset(Whox_diff, 0, sizeof(Dtype) * num_ * num_);
    memset(Whoy_diff, 0, sizeof(Dtype) * num_ * num_);
    const Dtype beta = 1.;
#ifdef _OPENMP
    omp_set_num_threads(threads_);
    #pragma omp parallel for
#endif
    for (int i = 0; i < bottom[bottom_id]->num(); ++i) {
      for (int x = end_x; end_x == 0 ? x <= start_x : x >= start_x; x -= inc_x) {
        for (int y = end_y; end_y == 0 ? y <= start_y : y >= start_y; y -= inc_y) {
          Dtype alpha_x = x != start_x ? -inc_x : 0.;
          Dtype alpha_y = y != start_y ? -inc_y : 0.;
#ifdef _OPENMP
    #pragma omp critical(whzx)
#endif
          strided_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_, num_, 1,
                  abs(alpha_x), bz_[bottom_id]->cpu_diff() + bz_[bottom_id]->offset(i, 0, x + alpha_x, y), ld_step,
                  bh_[bottom_id]->mutable_cpu_data() + bh_[bottom_id]->offset(i, 0, x, y), ld_step, beta,
                  Whzx_diff, num_);
#ifdef _OPENMP
    #pragma omp critical(whzy)
#endif
          strided_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_, num_, 1,
                  abs(alpha_y), bz_[bottom_id]->cpu_diff() + bz_[bottom_id]->offset(i, 0, x, y + alpha_y), ld_step,
                  bh_[bottom_id]->mutable_cpu_data() + bh_[bottom_id]->offset(i, 0, x, y), ld_step, beta,
                  Whzy_diff, num_);
#ifdef _OPENMP
    #pragma omp critical(whix)
#endif
          strided_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_, num_, 1,
                  abs(alpha_x), bi_[bottom_id]->cpu_diff() + bi_[bottom_id]->offset(i, 0, x + alpha_x, y), ld_step,
                  bh_[bottom_id]->mutable_cpu_data() + bh_[bottom_id]->offset(i, 0, x, y), ld_step, beta,
                  Whix_diff, num_);
#ifdef _OPENMP
    #pragma omp critical(whiy)
#endif
          strided_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_, num_, 1,
                  abs(alpha_y), bi_[bottom_id]->cpu_diff() + bi_[bottom_id]->offset(i, 0, x, y + alpha_y), ld_step,
                  bh_[bottom_id]->mutable_cpu_data() + bh_[bottom_id]->offset(i, 0, x, y), ld_step, beta,
                  Whiy_diff, num_);
#ifdef _OPENMP
    #pragma omp critical(whf1x)
#endif
          strided_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_, num_, 1,
                  abs(alpha_x), bf1_[bottom_id]->cpu_diff() + bf1_[bottom_id]->offset(i, 0, x + alpha_x, y), ld_step,
                  bh_[bottom_id]->mutable_cpu_data() + bh_[bottom_id]->offset(i, 0, x, y), ld_step, beta,
                  Whf1x_diff, num_);
#ifdef _OPENMP
    #pragma omp critical(whf1y)
#endif
          strided_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_, num_, 1,
                  abs(alpha_y), bf1_[bottom_id]->cpu_diff() + bf1_[bottom_id]->offset(i, 0, x, y + alpha_y), ld_step,
                  bh_[bottom_id]->mutable_cpu_data() + bh_[bottom_id]->offset(i, 0, x, y), ld_step, beta,
                  Whf1y_diff, num_);
#ifdef _OPENMP
    #pragma omp critical(whf2x)
#endif
          strided_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_, num_, 1,
                  abs(alpha_x), bf2_[bottom_id]->cpu_diff() + bf2_[bottom_id]->offset(i, 0, x + alpha_x, y), ld_step,
                  bh_[bottom_id]->mutable_cpu_data() + bh_[bottom_id]->offset(i, 0, x, y), ld_step, beta,
                  Whf2x_diff, num_);
#ifdef _OMPENMP
    #pragma omp critical(whf2y)
#endif
          strided_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_, num_, 1,
                  abs(alpha_y), bf2_[bottom_id]->cpu_diff() + bf2_[bottom_id]->offset(i, 0, x, y + alpha_y), ld_step,
                  bh_[bottom_id]->mutable_cpu_data() + bh_[bottom_id]->offset(i, 0, x, y), ld_step, beta,
                  Whf2y_diff, num_);
#ifdef _OPENMP
    #pragma omp critical(whox)
#endif
          strided_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_, num_, 1,
                  abs(alpha_x), bo_[bottom_id]->cpu_diff() + bo_[bottom_id]->offset(i, 0, x + alpha_x, y), ld_step,
                  bh_[bottom_id]->mutable_cpu_data() + bh_[bottom_id]->offset(i, 0, x, y), ld_step, beta,
                  Whox_diff, num_);
#ifdef _OPENMP
    #pragma omp critical(whoy)
#endif
          strided_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_, num_, 1,
                  abs(alpha_y), bo_[bottom_id]->cpu_diff() + bo_[bottom_id]->offset(i, 0, x, y + alpha_y), ld_step,
                  bh_[bottom_id]->mutable_cpu_data() + bh_[bottom_id]->offset(i, 0, x, y), ld_step, beta,
                  Whoy_diff, num_);
        }
      }
    }
    //Logging code for debug (to remove later)
    /*std::ofstream log_delta_top;
    std::ofstream log_delta_bh;
    std::ofstream log_delta_bo;
    std::ofstream log_delta_s;
    std::ofstream log_delta_bf2;
    std::ofstream log_delta_bi;
    std::ofstream log_delta_Whoy;
    std::ofstream log_delta_Whzx;

    log_delta_top.open("/home/snake91/Desktop/delta_top.txt");
    log_delta_bh.open("/home/snake91/Desktop/delta_bh.txt");
    log_delta_bo.open("/home/snake91/Desktop/delta_bo.txt");
    log_delta_s.open("/home/snake91/Desktop/delta_s.txt");
    log_delta_bf2.open("/home/snake91/Desktop/delta_bf2.txt");
    log_delta_bi.open("/home/snake91/Desktop/delta_bi.txt");
    log_delta_Whoy.open("/home/snake91/Desktop/delta_Whoy.txt");
    log_delta_Whzx.open("/home/snake91/Desktop/delta_Whzx.txt");

    const int count = top[0]->count();
    const int countW = num_ * num_;

    const Dtype *delta_top = top[0]->cpu_diff();
    const Dtype *delta_bh = bh_[0]->cpu_diff();
    const Dtype *delta_bo = bo_[0]->cpu_diff();
    const Dtype *delta_s = s_[0]->cpu_diff();
    const Dtype *delta_bf2 = bf2_[0]->cpu_diff();
    const Dtype *delta_bi = bi_[0]->cpu_diff();

    const Dtype *delta_Whoy = this->blobs_[9]->cpu_diff();
    const Dtype *delta_Whzx = this->blobs_[0]->cpu_diff();

    for (int i = 0; i < count; ++i) {
        log_delta_top << delta_top[i] << std::endl;
        log_delta_bh << delta_bh[i] << std::endl;
        log_delta_bo << delta_bo[i] << std::endl;
        log_delta_s << delta_s[i] << std::endl;
        log_delta_bf2 << delta_bf2[i] << std::endl;
        log_delta_bi << delta_bi[i] << std::endl;
    }
    for (int i = 0; i < countW; ++i) {
      log_delta_Whoy << delta_Whoy[i] << std::endl;
      log_delta_Whzx << delta_Whzx[i] << std::endl;
    }
    log_delta_top.close();
    log_delta_bh.close();
    log_delta_bo.close();
    log_delta_s.close();
    log_delta_bf2.close();
    log_delta_bi.close();
    log_delta_Whoy.close();
    log_delta_Whzx.close();
      string s;
      std::cin >> s;*/

    if (this->param_propagate_down_[bottom_id]) {

    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(MDLSTMLayer);
#endif

INSTANTIATE_CLASS(MDLSTMLayer);
REGISTER_LAYER_CLASS(MDLSTM);


}  // namespace caffe
