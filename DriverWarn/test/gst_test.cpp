#include "gstcamera.h"
#include "data_pool.h"
#include <thread>
#include "ai_inference.h"

using namespace ai2nference;
using namespace cv;

#define DEFAULT_AI_CONFIG "../res/ai_config.ini"
#define LANDMARK_PARSE_FILE "../models/landmark_contours.txt"

#define DEFAULT_GST_CONFIG "../res/gst_config.ini"

using namespace datapool;

std::shared_ptr<BufManager<cv::Mat>> rgb_object_frame;
std::vector<GstCamera *> gst_camera_vec;
std::vector <AiInference *> ai_inference_vec;

void getDataFromSample() {

    std::shared_ptr<GstSample> sample;
    for(;;) {
        if(gst_camera_vec.empty()) {
            DEBUG_FUNC();
            continue;
        }
        for(auto _camera : gst_camera_vec) {
            std::shared_ptr<GstSample> sample;
            GstCamera* pCam = _camera;
            static int count = 0;

            pCam->CaptureNoWait(sample);
            if(NULL == sample || NULL == sample.get()) {
                #ifdef DEBUG
                            std::cout << "the sample is null or invaild" << std::endl;
                #endif
                continue;
            }

            DEBUG_FUNC();
            GstCaps *sample_caps = gst_sample_get_caps(sample.get());
            if( sample_caps == NULL) {
                DEBUG_FUNC();
                continue;
            }
            // std::shared_ptr<cv::Mat> object_frame;
            gint sample_width,sample_height;

            GstStructure *structure = gst_caps_get_structure(sample_caps,0);
            gst_structure_get_int(structure,"width",&sample_width);
            gst_structure_get_int(structure,"height",&sample_height);
            DEBUG_FUNC();

            GstBuffer *gst_buffer = gst_sample_get_buffer(sample.get());
            if (NULL == gst_buffer || !sample_width || !sample_height) {
                continue;
            }
            cout <<"==sample width="<< sample_width <<",sample height = " << sample_height <<  endl;
            GstMapInfo sample_map;
            gst_buffer_map(gst_buffer,&sample_map,GST_MAP_READ);
            std::cout << "smaple map size: " << sample_map.size << std::endl;

            unsigned char *data_buffer = (unsigned char*)malloc(sizeof(unsigned char)*sample_map.size);
            if(data_buffer != nullptr) {
                DEBUG_FUNC();
                memset(data_buffer,0x00, sample_map.size);
                memcpy(data_buffer, (guchar *)sample_map.data, sample_map.size);
                DEBUG_FUNC();
                DataInfo data_info = {sample_width,sample_height,3,false,_camera->is_hwdec(),_camera->get_need_calibration()};
                cv::Mat tmp_mat = cv::Mat(sample_height,sample_width,CV_8UC4,data_buffer,0);
                std::shared_ptr<cv::Mat> object_frame =  std::make_shared<cv::Mat>(tmp_mat);
                rgb_object_frame->feed(object_frame);
                // stringstream str_name;
                // str_name << "record_" << count << ".png";
                // count++;
                // cv::imwrite(str_name.str(),tmp_mat);
                std::cout << __FILE__ << "================" << __LINE__ << std::endl;
            }
            free(data_buffer);
            //Avoiding memory leaks
            gst_buffer_unmap(gst_buffer, &sample_map);
            DEBUG_FUNC();
        }
    }
}


void ai2nfrence(){
    for(;;) {
        std::shared_ptr<cv::Mat> infe_imgframe;

        if(!rgb_object_frame) {
            DEBUG_FUNC();
            continue;
        }
        if(gst_camera_vec.empty() || ai_inference_vec.empty()) {
            DEBUG_FUNC();
            continue;
        }
        if(ai_inference_vec.size() != 3) {
            DEBUG_FUNC();
            continue;
        }
        if(rgb_object_frame->fetch()->empty()) {
            DEBUG_FUNC();
            continue;
        } else {
            DEBUG_FUNC();
            infe_imgframe = rgb_object_frame->fetch();
        }

        for (auto _inference : ai_inference_vec) {
            DEBUG_FUNC();
            std::vector<uchar> temp_packt = convertMat2Vector<uchar>(*infe_imgframe.get());
            if(_inference->get_ai_data_source() == "gst_zero") {
                ai_inference_vec[0]->loadTfliteData(1080,1920,3,temp_packt);
                std::vector<std::vector<float>> inference_result;
                ai_inference_vec[0]->runAndGetResult<float>(&inference_result);
                std::vector<cv::Point3_<float>> parse_results;
                DEBUG_FUNC();
                for(auto result : inference_result) {
                    std::cout << "result size " << result.size() << ",inferece result as follow:" << std::endl;
                    if(result.size() < 3) {
                        for(auto value : result){
                            std::cout << value << std::endl;
                        } 
                    } else {
                        for (int index = 0; index <  result.size()/3; ++index) {
                            parse_results.push_back(cv::Point3_<float>(result[3*index],result[3*index+1],result[3*index+2]));
                            std::cout << parse_results[index] << std::endl;
                        }
                    }
                }
            } else if(_inference->get_ai_data_source() == "gst_one") {
                DEBUG_FUNC();
                ai_inference_vec[2]->loadTfliteData(1080,1920,3,temp_packt);
                std::vector<std::vector<float>> inference_result;
                ai_inference_vec[2]->runAndGetResult<float>(&inference_result);
                DEBUG_FUNC();
                for(auto result : inference_result) {
                    std::cout << "inferece result as follow:" << std::endl;
                    for(auto value : result) {
                        std::cout << value;
                    }
                    std::cout << std::endl;
                }
            }
        }
    }
}

int main (int argc, char ** argv)
{
    MulitGstCamera::GstEnvInit();
    GMainLoop *main_loop = g_main_loop_new(NULL,false);
    IniConf ini_conf;
    memset(&ini_conf, 0, sizeof(IniConf));
    sprintf(ini_conf.ini_node, "conf_info");
    get_ini_info((char *)DEFAULT_GST_CONFIG,&ini_conf);

    rgb_object_frame = std::make_shared<BufManager<cv::Mat> > ();

    
    for(int i = 0; i < ini_conf.conf_count; i++) {
        std::cout << "gstreamer config info " << i << ",as follow:" <<std::endl;
        gst_camera_vec.push_back(new GstCamera(DEFAULT_GST_CONFIG,i));
    }

    datapool::DataPool <unsigned char> *data_pool = new DataPool<unsigned char>(1024,FALSE);
    for(auto gst_camera : gst_camera_vec) {
        gst_camera->Init();
        // std::thread gst_thread(&GstCamera::RunGst,gst_camera);
        std::thread gst_thread([=]{
            DEBUG_FUNC();
            gst_camera->RunGst(data_pool);
        });
        gst_thread.join();
    }

    memset(&ini_conf, 0, sizeof(IniConf));
    sprintf(ini_conf.ini_node, "conf_info");
    get_ini_info((char *)DEFAULT_AI_CONFIG,&ini_conf);

    for(int i = 0; i < ini_conf.conf_count; i++) {
        std::cout << "AI runtime config info " << i << ",as follow:" <<std::endl;
        ai_inference_vec.push_back(new AiInference(DEFAULT_AI_CONFIG,i));
    }

    for(auto ai_inference : ai_inference_vec) {
        std::cout << "loading tflite model ......" <<std::endl;
        ai_inference->loadTfliteModel();
    }
    DEBUG_FUNC();
    std::thread handleThread(getDataFromSample);
    handleThread.detach();

    std::this_thread::sleep_for(std::chrono::seconds(2));

    DEBUG_FUNC();
    std::thread inferenceThread(ai2nfrence);
    inferenceThread.detach();

    g_main_loop_run(main_loop);
    MulitGstCamera::GstEnvDeinit();
    g_main_loop_unref(main_loop);

#if 0
    // following code ,read param parse file include point info.
    std::vector<std::vector<int>> landmark_point_vec;
    {
        ifstream landmark_parse_file;
        string landmark_line;
        std::vector<string> landmark_str;
        std::vector<int> landmark_point;
        landmark_parse_file.open(LANDMARK_PARSE_FILE);
        getline(landmark_parse_file,landmark_line);
        while(landmark_parse_file && !landmark_line.empty()) {
            std::cout << "landmark line :" << landmark_line << std::endl;
            landmark_str = selfSplit(landmark_line," ");
            landmark_str.erase(std::begin(landmark_str));
            for(auto pos_value : landmark_str){
                landmark_point.push_back(std::stoi(pos_value));
            }
            sort(landmark_point.begin(),landmark_point.end());
            landmark_point.erase(unique(landmark_point.begin(), landmark_point.end()), landmark_point.end());
            landmark_point_vec.push_back(landmark_point);
            getline(landmark_parse_file,landmark_line);
        }
        landmark_parse_file.close();
    }

    static int count = 0;
    while(1){
        if(!data_pool->data_packet_vec.empty()) {

        } else {
            continue;
        }
        std::cout << __FILE__ << "================" << __LINE__ << std::endl;
        std::deque<datapool::DataPacket<unsigned char>*>::iterator packet_iter = data_pool->data_packet_vec.begin();
        std::deque<datapool::DataPacket<unsigned char>*>::iterator packet_iter_end = data_pool->data_packet_vec.end();
        for(;packet_iter != packet_iter_end;packet_iter++) {
            std::cout << "data pool size " <<  data_pool->data_packet_vec.size() << std::endl;
            std::cout << __FILE__ << "================" << __LINE__ << std::endl;

            std::cout << "data packet info:" << (*packet_iter)->data_info.channel << "," \
                << (*packet_iter)->data_info.width << "," << (*packet_iter)->data_info.height << std::endl;
            // cv::Mat tmp_mat = cv::Mat(*(*packet_iter)->data,CV_8UC4);
            if((*packet_iter)->source_name == "gst_zero") {
                std::vector<uchar> temp_packt = convertMat2Vector<uchar>((*packet_iter)->data_mat);
                std::thread ai_thread_1([=]{
                    ai_inference_vec[0]->loadTfliteData((*packet_iter)->data_mat.rows,(*packet_iter)->data_mat.cols,(*packet_iter)->data_mat.channels(),temp_packt);
                    std::vector<std::vector<float>> inference_result;
                    ai_inference_vec[0]->runAndGetResult<float>(&inference_result);
                    std::vector<cv::Point3_<float>> parse_results;
                    for(auto result : inference_result) {
                        std::cout << "result size " << result.size() << ",inferece result as follow:" << std::endl;
                        if(result.size() < 3) {
                            for(auto value : result){
                                std::cout << value << std::endl;
                            } 
                        } else {
                            for (int index = 0; index <  result.size()/3; ++index) {
                                parse_results.push_back(cv::Point3_<float>(result[3*index],result[3*index+1],result[3*index+2]));
                                std::cout << parse_results[index] << std::endl;
                            }
                        }
                    }
                    std::vector<cv::Point3_<float>> left_eye_point_vec;
                    std::vector<cv::Point3_<float>> right_eye_point_vec;
                    // get Site feature point
                    for(auto index : landmark_point_vec[1]) {
                        left_eye_point_vec.push_back(parse_results[index]);
                    }
                    for(auto index : landmark_point_vec[2]) {
                        right_eye_point_vec.push_back(parse_results[index]);
                    }

                    // get eye center point
                    cv::Point3f left_eye_center;
                    int left_point_num = left_eye_point_vec.size();
                    for(auto pos : left_eye_point_vec){
                        left_eye_center.x += pos.x;
                        left_eye_center.y += pos.y;
                        left_eye_center.z += pos.z;
                    }
                    left_eye_center.x = left_eye_center.x / left_point_num;
                    left_eye_center.y = left_eye_center.y / left_point_num;
                    left_eye_center.z = left_eye_center.z / left_point_num;

                    std::cout << "left eye center point :" << left_eye_center << std::endl;
                    cv::Mat source_mat = cv::imread("../models/img/face.jpeg",cv::COLOR_BGR2RGB);
                    cv::Mat tmp_mat;
                    cv::resize(source_mat,tmp_mat,cv::Size(192,192));
                    cv::Mat left_eye_mat = tmp_mat(cv::Rect2f(left_eye_center.x,left_eye_center.y,64,64));
                    cv::Mat con_left_eye_mat= left_eye_mat.clone();
                    vector<uchar> left_eye_vec = convertMat2Vector<uchar>(con_left_eye_mat);

                    ai_inference_vec[1]->loadTfliteData(left_eye_mat.rows,left_eye_mat.cols,left_eye_mat.channels(),left_eye_vec);
                    std::vector<std::vector<float>> left_eye_inference_result;
                    ai_inference_vec[1]->runAndGetResult<float>(&left_eye_inference_result);
                    for(auto result : left_eye_inference_result) {
                        std::cout << "inferece result as follow:" << std::endl;
                        for(auto value : result) {
                            std::cout << value;
                        }
                        std::cout << std::endl;
                    }
                });
                ai_thread_1.detach();
            } else if ((*packet_iter)->source_name == "gst_one") {
                std::vector<uchar> temp_packt = convertMat2Vector<uchar>((*packet_iter)->data_mat);
                std::thread ai_thread_2([=]{
                    ai_inference_vec[2]->loadTfliteData((*packet_iter)->data_mat.rows,(*packet_iter)->data_mat.cols,(*packet_iter)->data_mat.channels(),temp_packt);
                    std::vector<std::vector<float>> inference_result;
                    ai_inference_vec[2]->runAndGetResult<float>(&inference_result);
                    for(auto result : inference_result) {
                        std::cout << "inferece result as follow:" << std::endl;
                        for(auto value : result) {
                            std::cout << value;
                        }
                        std::cout << std::endl;
                    }
                });
                ai_thread_2.detach();
            }

            // cv::Mat tmp_mat = (*packet_iter)->data_mat;
            // std::cout << "gst test mat channels,rows,cols:" << tmp_mat.channels() << "," << tmp_mat.rows << "," << tmp_mat.cols << std::endl;;
            // // cv::Mat result_mat = tmp_mat.reshape(4,(*packet_iter)->data_info.height).clone();
            // stringstream str_name;
            // str_name << "./gst_record_" << count++ << ".jpg";
            // // cv::imwrite(str_name.str(),result_mat);
            // cv::imwrite(str_name.str(),tmp_mat);
            // std::cout << __FILE__ << "================" << __LINE__ << std::endl;
            // // (*packet_iter)->~DataPacket();
            // std::cout << __FILE__ << "================" << __LINE__ << std::endl;
            std::unique_lock<std::mutex> locker(data_pool->queue_mutex);
            data_pool->condition.wait(locker);
            // data_pool->rmPackage((*packet_iter));
            (*packet_iter)->data_mat.~Mat();
            data_pool->data_packet_vec.erase(packet_iter);
            // std::deque<datapool::DataPacket<unsigned char>*>(data_pool->data_packet_vec).swap(data_pool->data_packet_vec);
            if(!data_pool->data_packet_vec.empty()) {
                locker.unlock();
                packet_iter = data_pool->data_packet_vec.begin();
                packet_iter_end = data_pool->data_packet_vec.end();
                std::cout << __FILE__ << "================" << __LINE__ << std::endl;
            } else {
                locker.unlock();
                std::cout << __FILE__ << "================" << __LINE__ << std::endl;
                break;
            }
        }
    }
#endif

}