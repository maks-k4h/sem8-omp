#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include <omp.h>

using namespace cv;

struct Cluster {
    float sum[5] = {0};
    int count = 0;
};

int main() {
    const int k = 15;         // Number of clusters
    const int max_iters = 100;
    const float epsilon = 1.0f;
    const std::string input_path = "input.png";
    const std::string output_path = "output.png";

    // Read and prepare image
    Mat image = imread(input_path);
    if (image.empty()) {
        std::cerr << "Error: Could not read image" << std::endl;
        return 1;
    }
    cvtColor(image, image, COLOR_BGR2RGB);
    
    const int rows = image.rows;
    const int cols = image.cols;
    const int total_pixels = rows * cols;
    
    // Create feature vectors [R, G, B, X, Y]
    std::vector<float> features(total_pixels * 5);
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            int idx = y * cols + x;
            Vec3b color = image.at<Vec3b>(y, x);
            features[idx*5 + 0] = color[0];
            features[idx*5 + 1] = color[1];
            features[idx*5 + 2] = color[2];
            features[idx*5 + 3] = x;
            features[idx*5 + 4] = y;
        }
    }

    // Initialize centroids randomly
    std::vector<float> centroids(k * 5);
    srand(12345);
    for (int i = 0; i < k; ++i) {
        int idx = rand() % total_pixels;
        for (int j = 0; j < 5; ++j) {
            centroids[i*5 + j] = features[idx*5 + j];
        }
    }

    // K-means iteration
    std::vector<int> labels(total_pixels);
    bool converged = false;
    int iter = 0;
    
    while (!converged && iter < max_iters) {
        std::vector<Cluster> clusters(k);
        float max_diff = 0.0f;

        // Assign points to clusters
        #pragma omp parallel for reduction(max:max_diff)
        for (int i = 0; i < total_pixels; ++i) {
            float min_dist = INFINITY;
            int best_cluster = -1;
            float* pixel = &features[i*5];
            
            // Find nearest cluster
            for (int c = 0; c < k; ++c) {
                float dist = 0.0f;
                for (int j = 0; j < 5; ++j) {
                    float diff = pixel[j] - centroids[c*5 + j];
                    dist += diff * diff;
                }
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = c;
                }
            }
            
            // Update cluster sums
            #pragma omp critical
            {
                for (int j = 0; j < 5; ++j) {
                    clusters[best_cluster].sum[j] += pixel[j];
                }
                clusters[best_cluster].count++;
            }
            
            labels[i] = best_cluster;
        }

        // Update centroids and check convergence
        #pragma omp parallel for reduction(max:max_diff)
        for (int c = 0; c < k; ++c) {
            if (clusters[c].count == 0) continue;
            
            float new_centroid[5];
            for (int j = 0; j < 5; ++j) {
                new_centroid[j] = clusters[c].sum[j] / clusters[c].count;
            }
            
            // Calculate centroid movement
            float diff = 0.0f;
            for (int j = 0; j < 5; ++j) {
                float d = new_centroid[j] - centroids[c*5 + j];
                diff += d * d;
            }
            diff = std::sqrt(diff);
            
            if (diff > max_diff) max_diff = diff;
            
            // Update centroids
            for (int j = 0; j < 5; ++j) {
                centroids[c*5 + j] = new_centroid[j];
            }
        }

        converged = (max_diff < epsilon);
        iter++;
    }

    // Create segmented image
    Mat segmented(rows, cols, CV_8UC3);
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            int idx = y * cols + x;
            int cluster = labels[idx];
            segmented.at<Vec3b>(y, x) = Vec3b(
                static_cast<uchar>(centroids[cluster*5 + 0]),
                static_cast<uchar>(centroids[cluster*5 + 1]),
                static_cast<uchar>(centroids[cluster*5 + 2])
            );
        }
    }

    cvtColor(segmented, segmented, COLOR_RGB2BGR);
    imwrite(output_path, segmented);
    std::cout << "Segmentation complete. Saved to " << output_path << std::endl;

    return 0;
}