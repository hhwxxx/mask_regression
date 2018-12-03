#include <algorithm>
#include <cfloat>
#include <vector>
#include <fstream>
#include <iomanip>

#include "caffe/layers/iou_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
	//sigmoid function
	//template <typename Dtype>
	//inline Dtype sigmoid(Dtype x) {
	//	return 1. / (1. + exp(-x));
	//}

	template <typename Dtype>
	void IouLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		LossLayer<Dtype>::LayerSetUp(bottom, top);

		IouLossParameter iou_loss_param_ = this->layer_param_.iou_loss_param();
		lamda_ = iou_loss_param_.lamda();
	}

	template <typename Dtype>
	void IouLossLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		LossLayer<Dtype>::Reshape(bottom, top);

		CHECK_EQ(bottom[0]->num(), bottom[1]->num())
			<< "The data and label should have the same number.";

		CHECK_EQ(bottom.size(), 4)
			<< "must have 4 bottom blobs. ";

		vector<int> loss_shape(1,1); 
		top[0]->Reshape(loss_shape);

		vector<int> aabb_shape(4, 1);
		aabb_shape[0] = bottom[0]->num();
		aabb_shape[1] = 1;
		aabb_shape[2] = bottom[0]->height();
		aabb_shape[3] = bottom[0]->width();
		aabb_target.Reshape(aabb_shape);
		theta_target.Reshape(aabb_shape);

		iou.Reshape(aabb_shape);
		delta_theta.Reshape(aabb_shape);
		inter_area.Reshape(aabb_shape);
		union_area.Reshape(aabb_shape);
		bdiff.ReshapeLike(*bottom[0]);
	}

	template <typename Dtype>
	void IouLossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
			
		//string predname = "/home/wangjie/wangjie/Experiment/East/pred.txt";
		//std::ofstream out_file(predname);
		//int width = bottom[0]->width();
		//int height = bottom[0]->height();
		//int channel = bottom[0]->channels();
		////cv::Mat pred_mat(height, width, CV_32FC1);
		//int index = 0;
		//for (int c = 0; c < channel; c++)
		//{
		//for (int h = 0; h < height; h++)
		//{
		//	//float *ptr_row = pred_mat.ptr<float>(h);
		//	for (int w = 0; w < width; w++)
		//	{
		//		//double a = bottom[0]->cpu_data()[index];
		//		out_file << std::setprecision(20) << bottom[0]->cpu_data()[index++] << " ";
		//		//ptr_row[w] = bottom[0]->cpu_data()[index++];
		//	}
		//	out_file << std::endl;
		//}
		//}
		//out_file.close();
		//
		//predname = "/home/wangjie/wangjie/Experiment/East/label.txt";
		//std::ofstream out_file1(predname);
		// width = bottom[1]->width();
		// height = bottom[1]->height();
		////cv::Mat pred_mat(height, width, CV_32FC1);
		// index = 0;
		// for (int c = 0; c < channel; c++)
		//{
		//for (int h = 0; h < height; h++)
		//{
		//	//float *ptr_row = pred_mat.ptr<float>(h);
		//	for (int w = 0; w < width; w++)
		//	{
		//		//double a = bottom[0]->cpu_data()[index];
		//		out_file1 << std::setprecision(20) << bottom[1]->cpu_data()[index++] << " ";
		//		//ptr_row[w] = bottom[0]->cpu_data()[index++];
		//	}
		//	out_file1 << std::endl;
		//}
		//}
		//out_file1.close();
		//
		// predname = "/home/wangjie/wangjie/Experiment/East/mask.txt";
		//std::ofstream out_file2(predname);
		// width = bottom[2]->width();
		// height = bottom[2]->height();
		////cv::Mat pred_mat(height, width, CV_32FC1);
		// index = 0;
		//for (int h = 0; h < height; h++)
		//{
		//	//float *ptr_row = pred_mat.ptr<float>(h);
		//	for (int w = 0; w < width; w++)
		//	{
		//		//double a = bottom[0]->cpu_data()[index];
		//		out_file2 << std::setprecision(20) << bottom[2]->cpu_data()[index++] << " ";
		//		//ptr_row[w] = bottom[0]->cpu_data()[index++];
		//	}
		//	out_file2 << std::endl;
		//}
		//out_file2.close();
		//
		//predname = "/home/wangjie/wangjie/Experiment/East/conf.txt";
		//std::ofstream out_file3(predname);
		// width = bottom[3]->width();
		// height = bottom[3]->height();
		////cv::Mat pred_mat(height, width, CV_32FC1);
		// index = 0;
		//for (int h = 0; h < height; h++)
		//{
		//	//float *ptr_row = pred_mat.ptr<float>(h);
		//	for (int w = 0; w < width; w++)
		//	{
		//		//double a = bottom[0]->cpu_data()[index];
		//		out_file3 << std::setprecision(20) << bottom[3]->cpu_data()[index++] << " ";
		//		//ptr_row[w] = bottom[0]->cpu_data()[index++];
		//	}
		//	out_file3 << std::endl;
		//}
		//out_file3.close();

		const Dtype* loc_pred = bottom[0]->cpu_data();
		const Dtype* loc_label = bottom[1]->cpu_data();
		const Dtype* loc_mask = bottom[2]->cpu_data();
		const Dtype* loc_conf = bottom[3]->cpu_data();

		int num = bottom[2]->num();
		int inner_dim = bottom[2]->height()*bottom[2]->width();

		Dtype *aabb_loss = aabb_target.mutable_cpu_data();
		Dtype *theta_loss = theta_target.mutable_cpu_data();

		Dtype *aabb_up = inter_area.mutable_cpu_data();
		Dtype *aabb_down = union_area.mutable_cpu_data();
		Dtype *theta_diff = delta_theta.mutable_cpu_data();
		Dtype *aabb_iou = iou.mutable_cpu_data();

		const int width_ = bottom[0]->width();
		const int height_ = bottom[0]->height();
		const int channels = bottom[0]->channels();
		int counsum = 0;

		for (int i = 0; i < num; i++)
		{
			int offset = i*inner_dim * channels;
			for (int h = 0; h < height_; h++)
			{
				for (int w = 0; w < width_; w++)
				{
					Dtype mask = loc_mask[i*inner_dim + h*width_ + w];
					Dtype conf = loc_conf[i*inner_dim + h*width_ + w];
					if (mask < 1e-3)
					{
						aabb_loss[i*inner_dim + h*width_ + w] = 0;
						theta_loss[i*inner_dim + h*width_ + w] = 0;
					}else if (conf < 0.5)
					{
						aabb_loss[i*inner_dim + h*width_ + w] = 0;
						theta_loss[i*inner_dim + h*width_ + w] = 0;
					}
					else
					{
						counsum+=1;
						Dtype h1_pred = loc_pred[offset + h*width_ + w];  //up
						Dtype w1_pred = loc_pred[offset + inner_dim + h*width_ + w]; //right
						Dtype h2_pred = loc_pred[offset + 2 * inner_dim + h*width_ + w];  //down
						Dtype w2_pred = loc_pred[offset + 3 * inner_dim + h*width_ + w]; //left
						Dtype theta_pred = loc_pred[offset + 4 * inner_dim + h*width_ + w]; //left
						

						Dtype h1_label = loc_label[offset + h*width_ + w];  //up
						Dtype w1_label = loc_label[offset + inner_dim + h*width_ + w]; //right
						Dtype h2_label = loc_label[offset + 2 * inner_dim + h*width_ + w];  //down
						Dtype w2_label = loc_label[offset + 3 * inner_dim + h*width_ + w]; //left
						Dtype theta_label = loc_label[offset + 4 * inner_dim + h*width_ + w]; //left



						//aabb iou 
						Dtype aabb = 1 + (std::min(h1_pred, h1_label) + std::min(h2_pred, h2_label))*(std::min(w1_pred,w1_label)+std::min(w2_pred,w2_label));
						Dtype union_val = 2 + (h1_pred + h2_pred)*(w1_pred + w2_pred) + (h1_label + h2_label)*(w1_label + w2_label) - aabb;

						aabb_up[i*inner_dim + h*width_ + w] = aabb;
						aabb_down[i*inner_dim + h*width_ + w] = union_val;
						Dtype iou_ = (aabb / union_val);
						aabb_iou[i*inner_dim + h*width_ + w] = iou_;
						aabb_loss[i*inner_dim + h*width_ + w] = -1.*log(iou_);
						if (-1.*log(iou_) != -1.*log(iou_) || abs(-1.*log(iou_)) > 1000)
						{
							LOG(INFO) << iou_;
						LOG(INFO) << h1_pred << ' ' << h1_label;
						LOG(INFO) << w1_pred << ' ' << w1_label;
						LOG(INFO) << h2_pred << ' ' << h2_label;
						LOG(INFO) << w2_pred << ' ' << w2_label;


						}

						//theta
						Dtype theta_diff_ = (theta_pred - theta_label);
						theta_diff[i*inner_dim + h*width_ + w] = theta_diff_;
						theta_loss[i*inner_dim + h*width_ + w] = 1. - cos(theta_diff_);
					}
				}
			}
		}

		int count = num*inner_dim;
		//LOG(INFO) << caffe_cpu_asum(count, aabb_target.cpu_data());
		//LOG(INFO) << counsum;
		//Dtype iou_loss_ = caffe_cpu_asum(count, aabb_target.cpu_data()) / (counsum + 1);
		//Dtype theta_loss_ = caffe_cpu_asum(count, theta_target.cpu_data()) / (counsum + 1);
		Dtype iou_loss_ = caffe_cpu_asum(count, aabb_target.cpu_data());
		Dtype theta_loss_ = caffe_cpu_asum(count, theta_target.cpu_data());
		//LOG(INFO) << iou_loss_/count;
		//LOG(INFO) << theta_loss_/count;

		top[0]->mutable_cpu_data()[0] = (iou_loss_ + lamda_*theta_loss_)/count;
		//LOG(INFO) << top[0]->mutable_cpu_data()[0];
	}



	template <typename Dtype>
	void IouLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		if (propagate_down[1] || propagate_down[2]) {
			LOG(FATAL) << this->type()
				<< " Layer cannot backpropagate to label inputs.";
		}
		if (propagate_down[0]) {
			const Dtype* loc_pred = bottom[0]->cpu_data();
			const Dtype* loc_label = bottom[1]->cpu_data();
			const Dtype* loc_mask = bottom[2]->cpu_data();
			const Dtype* loc_conf = bottom[3]->cpu_data();

			Dtype* loc_diff = bdiff.mutable_cpu_diff();
			//Dtype* aabb_diff = bottom[1]->mutable_cpu_diff();
			//Dtype* theta_diff = bottom[2]->mutable_cpu_diff();

			Dtype *aabb_iou = iou.mutable_cpu_data();
			Dtype *aabb_up = inter_area.mutable_cpu_data();
			Dtype *aabb_down = union_area.mutable_cpu_data();
			Dtype *theta_diff = delta_theta.mutable_cpu_data();

			int inner_dim = bottom[2]->height()*bottom[2]->width();
			const int num_ = bottom[0]->num();
			const int width_ = bottom[0]->width();
			const int height_ = bottom[0]->height();
			const int channels = bottom[0]->channels();
			int counsum = 0;

			for (int i = 0; i < num_; i++){
				int offset = i*inner_dim * channels;
				for (int h = 0; h < height_; ++h){
					for (int w = 0; w < width_; ++w) {
						Dtype mask = loc_mask[i*inner_dim + h*width_ + w];
						Dtype conf = loc_conf[i*inner_dim + h*width_ + w];
						if (mask < 1e-3)
						{
							loc_diff[offset + h*width_ + w] = 0;
							loc_diff[offset + inner_dim + h*width_ + w] = 0;
							loc_diff[offset + 2*inner_dim + h*width_ + w] = 0;
							loc_diff[offset + 3*inner_dim + h*width_ + w] = 0;
							loc_diff[offset + 4*inner_dim + h*width_ + w] = 0;
						}else if (conf < 0.5)
						{
							loc_diff[offset + h*width_ + w] = 0;
							loc_diff[offset + inner_dim + h*width_ + w] = 0;
							loc_diff[offset + 2 * inner_dim + h*width_ + w] = 0;
							loc_diff[offset + 3 * inner_dim + h*width_ + w] = 0;
							loc_diff[offset + 4 * inner_dim + h*width_ + w] = 0;
						}
						else
						{
							counsum += 1;
							Dtype h1_pred = loc_pred[offset + h*width_ + w];  //up
							Dtype w1_pred = loc_pred[offset + inner_dim + h*width_ + w]; //right
							Dtype h2_pred = loc_pred[offset + 2 * inner_dim + h*width_ + w];  //down
							Dtype w2_pred = loc_pred[offset + 3 * inner_dim + h*width_ + w]; //left
							Dtype theta_pred = loc_pred[offset + 4 * inner_dim + h*width_ + w]; //left

							Dtype h1_label = loc_label[offset + h*width_ + w];  //up
							Dtype w1_label = loc_label[offset + inner_dim + h*width_ + w]; //right
							Dtype h2_label = loc_label[offset + 2 * inner_dim + h*width_ + w];  //down
							Dtype w2_label = loc_label[offset + 3 * inner_dim + h*width_ + w]; //left
							Dtype theta_label = loc_label[offset + 4 * inner_dim + h*width_ + w]; //left

							//aabb iou 
							if (h1_pred >= h1_label)
							{
								Dtype alpha = 1. / aabb_iou[i*inner_dim + h*width_ + w];
								Dtype derive = (w1_pred + w2_pred)*aabb_up[i*inner_dim + h*width_ + w] / (aabb_down[i*inner_dim + h*width_ + w] * aabb_down[i*inner_dim + h*width_ + w]);
								loc_diff[offset + h*width_ + w] = alpha*derive;
							}
							else{
								Dtype alpha = -1. / aabb_iou[i*inner_dim + h*width_ + w];
								Dtype derive = (std::min(w1_label, w1_pred) + std::min(w2_label, w2_pred))*aabb_down[i*inner_dim + h*width_ + w] - (w1_pred + w2_pred - (std::min(w1_label, w1_pred) + std::min(w2_label, w2_pred)))*aabb_up[i*inner_dim + h*width_ + w];
								derive /= (aabb_down[i*inner_dim + h*width_ + w] * aabb_down[i*inner_dim + h*width_ + w]);
								loc_diff[offset + h*width_ + w] = alpha*derive;
							}

							if (h2_pred >= h2_label)
							{
								Dtype alpha = 1. / aabb_iou[i*inner_dim + h*width_ + w];
								Dtype derive = (w1_pred + w2_pred)*aabb_up[i*inner_dim + h*width_ + w] / (aabb_down[i*inner_dim + h*width_ + w] * aabb_down[i*inner_dim + h*width_ + w]);
								loc_diff[offset + 2*inner_dim + h*width_ + w] = alpha*derive;
							}
							else{
								Dtype alpha = -1. / aabb_iou[i*inner_dim + h*width_ + w];
								Dtype derive = (std::min(w1_label, w1_pred) + std::min(w2_label, w2_pred))*aabb_down[i*inner_dim + h*width_ + w] - (w1_pred + w2_pred - (std::min(w1_label, w1_pred) + std::min(w2_label, w2_pred)))*aabb_up[i*inner_dim + h*width_ + w];
								derive /= (aabb_down[i*inner_dim + h*width_ + w] * aabb_down[i*inner_dim + h*width_ + w]);
								loc_diff[offset + 2*inner_dim + h*width_ + w] = alpha*derive;
							}

							if (w1_pred >= w1_label)
							{
								Dtype alpha = 1. / aabb_iou[i*inner_dim + h*width_ + w];
								Dtype derive = (h1_pred + h2_pred)*aabb_up[i*inner_dim + h*width_ + w] / (aabb_down[i*inner_dim + h*width_ + w] * aabb_down[i*inner_dim + h*width_ + w]);
								loc_diff[offset + inner_dim + h*width_ + w] = alpha*derive;
							}
							else{
								Dtype alpha = -1. / aabb_iou[i*inner_dim + h*width_ + w];
								Dtype derive = (std::min(h1_label, h1_pred) + std::min(h2_label, h2_pred))*aabb_down[i*inner_dim + h*width_ + w] - (h1_pred + h2_pred - (std::min(h1_label, h1_pred) + std::min(h2_label, h2_pred)))*aabb_up[i*inner_dim + h*width_ + w];
								derive /= (aabb_down[i*inner_dim + h*width_ + w] * aabb_down[i*inner_dim + h*width_ + w]);
								loc_diff[offset + inner_dim + h*width_ + w] = alpha*derive;
							}

							if (w2_pred >= w2_label)
							{
								Dtype alpha = 1. / aabb_iou[i*inner_dim + h*width_ + w];
								Dtype derive = (h1_pred + h2_pred)*aabb_up[i*inner_dim + h*width_ + w] / (aabb_down[i*inner_dim + h*width_ + w] * aabb_down[i*inner_dim + h*width_ + w]);
								loc_diff[offset + 3*inner_dim + h*width_ + w] = alpha*derive;
							}
							else{
								Dtype alpha = -1. / aabb_iou[i*inner_dim + h*width_ + w];
								Dtype derive = (std::min(h1_label, h1_pred) + std::min(h2_label, h2_pred))*aabb_down[i*inner_dim + h*width_ + w] - (h1_pred + h2_pred - (std::min(h1_label, h1_pred) + std::min(h2_label, h2_pred)))*aabb_up[i*inner_dim + h*width_ + w];
								derive /= (aabb_down[i*inner_dim + h*width_ + w] * aabb_down[i*inner_dim + h*width_ + w]);
								loc_diff[offset + 3*inner_dim + h*width_ + w] = alpha*derive;
							}

							//theta
							loc_diff[offset + 4 * inner_dim + h*width_ + w] = lamda_*(sin(theta_diff[i*inner_dim + h*width_ + w]));
						}
					}
				}
			}
			int count = num_ * inner_dim;
			//Dtype alpha = top[0]->cpu_diff()[0]/(counsum + 1);
			Dtype alpha = top[0]->cpu_diff()[0]/count;

			//for (int haha =0; haha<bottom[0]->count(); haha++)
			//{
			//	if (bdiff.cpu_diff()[haha] > 0)
			//	{
			//		LOG(INFO) << bdiff.cpu_diff()[haha];
			//	}
			//}
			caffe_cpu_axpby(bottom[0]->count(), alpha, bdiff.cpu_diff(), Dtype(0), bottom[0]->mutable_cpu_diff());
			//for (int haha =0; haha<bottom[0]->count(); haha++)
			//{
			//	if (bottom[0]->cpu_diff()[haha] > 0)
			//	{
			//		LOG(INFO) << bottom[0]->cpu_diff()[haha];
			//	}
			//}
			//string diffname = "/home/wangjie/wangjie/Experiment/East/diff.txt";
			//std::ofstream out_file(diffname);
			//int width = bottom[0]->width();
			//int height = bottom[0]->height();
			//int channel = bottom[0]->channels();
			////cv::Mat pred_mat(height, width, CV_32FC1);
			//int index = 0;
			//for (int c = 0; c < channel; c++)
			//{
			//for (int h = 0; h < height; h++)
			//{
			//	//float *ptr_row = pred_mat.ptr<float>(h);
			//	for (int w = 0; w < width; w++)
			//	{
			//		//double a = bottom[0]->cpu_data()[index];
			//		out_file << std::setprecision(20) << bottom[0]->cpu_diff()[index++] << " ";
			//		//ptr_row[w] = bottom[0]->cpu_data()[index++];
			//	}
			//	out_file << std::endl;
			//}
			//}
			//out_file.close();
			
		}
	}
//#ifdef CPU_ONLY
//	STUB_GPU(TextLossLayer);
//#endif

	INSTANTIATE_CLASS(IouLossLayer);
	REGISTER_LAYER_CLASS(IouLoss);

}  // namespace caffe

