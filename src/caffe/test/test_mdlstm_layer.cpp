#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class MDLSTMLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  MDLSTMLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 10, 2, 3)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(2.);
    ConstantFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MDLSTMLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MDLSTMLayerTest, TestDtypesAndDevices);

TYPED_TEST(MDLSTMLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MDLSTMParameter* mdlstm_param =
      layer_param.mutable_mdlstm_param();
  mdlstm_param->set_num_hidden(2);
  shared_ptr<MDLSTMLayer<Dtype> > layer(
      new MDLSTMLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 3);
  EXPECT_EQ(this->blob_top_->channels(), 2);
}

TYPED_TEST(MDLSTMLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    MDLSTMParameter* mdlstm_param =
        layer_param.mutable_mdlstm_param();
    mdlstm_param->set_num_hidden(2);
    mdlstm_param->mutable_weight_filler()->set_type("constant");
    mdlstm_param->mutable_weight_filler()->set_value(3.);
    shared_ptr<MDLSTMLayer<Dtype> > layer(
        new MDLSTMLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    const Dtype* data = this->blob_top_->cpu_data();

    EXPECT_NEAR(0.6082834, data[0], 10E-05);
    EXPECT_NEAR(0.9477099, data[1], 10E-05);
    EXPECT_NEAR(0.9927579, data[2], 10E-05);
    EXPECT_NEAR(0.9477099, data[3], 10E-05);
    EXPECT_NEAR(0.9998280, data[4], 10E-05);
    EXPECT_NEAR(0.9999990, data[5], 10E-05);
    EXPECT_NEAR(0.6082834, data[6], 10E-05);
    EXPECT_NEAR(0.9477099, data[7], 10E-05);
    EXPECT_NEAR(0.9927579, data[8], 10E-05);
    EXPECT_NEAR(0.9477099, data[9], 10E-05);
    EXPECT_NEAR(0.9998280, data[10], 10E-05);
    EXPECT_NEAR(0.9999990, data[11], 10E-05);
    EXPECT_NEAR(0.6082834, data[12], 10E-05);
    EXPECT_NEAR(0.9477099, data[13], 10E-05);
    EXPECT_NEAR(0.9927579, data[14], 10E-05);
    EXPECT_NEAR(0.9477099, data[15], 10E-05);
    EXPECT_NEAR(0.9998280, data[16], 10E-05);
    EXPECT_NEAR(0.9999990, data[17], 10E-05);
    EXPECT_NEAR(0.6082834, data[18], 10E-05);
    EXPECT_NEAR(0.9477099, data[19], 10E-05);
    EXPECT_NEAR(0.9927579, data[20], 10E-05);
    EXPECT_NEAR(0.9477099, data[21], 10E-05);
    EXPECT_NEAR(0.9998280, data[22], 10E-05);
    EXPECT_NEAR(0.9999990, data[23], 10E-05);

  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(MDLSTMLayerTest, TestGradient1) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    MDLSTMParameter* mdlstm_param =
        layer_param.mutable_mdlstm_param();
    mdlstm_param->set_num_hidden(2);
    mdlstm_param->mutable_weight_filler()->set_type("gaussian");
//    mdlstm_param->mutable_weight_filler()->set_value(3.);
//    mdlstm_param->set_vertical_dir(MDLSTMParameter_VerticalDirection_DOWN);
//    mdlstm_param->set_horizontal_dir(MDLSTMParameter_HorizontalDirection_RIGHT);
    MDLSTMLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(MDLSTMLayerTest, TestGradient2) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    MDLSTMParameter* mdlstm_param =
            layer_param.mutable_mdlstm_param();
    mdlstm_param->set_num_hidden(2);
    mdlstm_param->mutable_weight_filler()->set_type("gaussian");
    // mdlstm_param->mutable_weight_filler()->set_value(3.);
    mdlstm_param->set_vertical_dir(MDLSTMParameter_VerticalDirection_UP);
    mdlstm_param->set_horizontal_dir(MDLSTMParameter_HorizontalDirection_LEFT);
    MDLSTMLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                    this->blob_top_vec_);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(MDLSTMLayerTest, TestGradient3) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    MDLSTMParameter* mdlstm_param =
            layer_param.mutable_mdlstm_param();
    mdlstm_param->set_num_hidden(2);
    mdlstm_param->mutable_weight_filler()->set_type("gaussian");
    // mdlstm_param->mutable_weight_filler()->set_value(3.);
    mdlstm_param->set_vertical_dir(MDLSTMParameter_VerticalDirection_DOWN);
    mdlstm_param->set_horizontal_dir(MDLSTMParameter_HorizontalDirection_LEFT);
    MDLSTMLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                    this->blob_top_vec_);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(MDLSTMLayerTest, TestGradient4) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    MDLSTMParameter* mdlstm_param =
            layer_param.mutable_mdlstm_param();
    mdlstm_param->set_num_hidden(2);
    mdlstm_param->mutable_weight_filler()->set_type("gaussian");
    // mdlstm_param->mutable_weight_filler()->set_value(3.);
    mdlstm_param->set_vertical_dir(MDLSTMParameter_VerticalDirection_UP);
    mdlstm_param->set_horizontal_dir(MDLSTMParameter_HorizontalDirection_RIGHT);
    MDLSTMLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                    this->blob_top_vec_);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}


}  // namespace caffe
