
#include "popsift.h"

namespace pps{

PopSiftContext *ctx = nullptr;

PopSiftContext::PopSiftContext() : ps(nullptr){
    cudaDeviceReset();

    popsift::cuda::device_prop_t deviceInfo;
    deviceInfo.set(0, false);
}

PopSiftContext::~PopSiftContext(){
    ps->uninit();
    delete ps;
    ps = nullptr;
}

void PopSiftContext::setup(float peak_threshold, float edge_threshold){
    bool changed = false;
    if (this->peak_threshold != peak_threshold) { this->peak_threshold = peak_threshold; changed = true; }
    if (this->edge_threshold != edge_threshold) { this->edge_threshold = edge_threshold; changed = true; }
    
    if (changed){
        config.setThreshold(peak_threshold);
        config.setEdgeLimit(edge_threshold);

        // TODO: root sift config
        //config.setNormMode(rootSift ? popsift::Config::RootSift : popsift::Config::Classic);
        config.setFilterSorting(popsift::Config::LargestScaleFirst);
        // TODO: more?

        // Rebuild ps object
        if (ps){
            ps->uninit();
            delete ps;
        }
        ps = new PopSift(config,
                    popsift::Config::ProcessingMode::ExtractingMode,
                    PopSift::FloatImages );
    }
}

PopSift *PopSiftContext::get(){
    return ps;
}

py::object popsift(pyarray_f image,
                 float peak_threshold,
                 float edge_threshold,
                 int target_num_features) {
    py::gil_scoped_release release;

    if (!image.size()) return py::none();

    if (!ctx) ctx = new PopSiftContext();

    int width = image.shape(1);
    int height = image.shape(0);
    int numFeatures = 0;
    
    while(true){
        ctx->setup(peak_threshold, edge_threshold);
        std::unique_ptr<SiftJob> job(ctx->get()->enqueue( width, height, image.data() ));
        std::unique_ptr<popsift::Features> result(job->get());
        numFeatures = result->getFeatureCount();

        // std::cerr << "Number of feature points: " << result->getFeatureCount()
        //      << " number of feature descriptors: " << result->getDescriptorCount()
        //      << std::endl;

        if (numFeatures >= target_num_features || peak_threshold < 0.0001){
            popsift::Feature* feature_list = result->getFeatures();
            std::vector<float> points(4 * numFeatures);
            std::vector<float> desc(128 * numFeatures);

            for (size_t i = 0; i < numFeatures; i++){
                popsift::Feature pFeat = feature_list[i];

                for(int oriIdx = 0; oriIdx < pFeat.num_ori; oriIdx++){
                    const popsift::Descriptor* pDesc = pFeat.desc[oriIdx];

                    for (int k = 0; k < 128; k++){
                        desc[128 * i + k] = pDesc->features[k];
                    }

                    points[4 * i + 0] = pFeat.xpos;
                    points[4 * i + 1] = pFeat.ypos;
                    points[4 * i + 2] = pFeat.sigma;
                    points[4 * i + 3] = pFeat.orientation[oriIdx];
                }
            }

            py::list retn;
            retn.append(py_array_from_data(&points[0], numFeatures, 4));
            retn.append(py_array_from_data(&desc[0], numFeatures, 128));
            return retn;
        }else{
            // Lower peak threshold if we don't meet the target
            peak_threshold = (peak_threshold * 2.0) / 3.0;
        }
    }

    // We should never get here
    return py::none();
}

}