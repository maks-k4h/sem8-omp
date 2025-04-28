#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <omp.h>

using namespace cv;

int main(int argc, char** argv) {
    int k = 5;         // Number of clusters
    int max_iters = 100;
    float epsilon = 1.0f;
    
    Mat image = imread("input.png");
    if (image.empty()) {
        std::cerr << "Could not read the image." << std::endl;
        return 1;
    }
    
    cvtColor(image, image, COLOR_BGR2RGB);
    const int rows = image.rows;
    const int cols = image.cols;
    const int total_pixels = rows * cols;
    
    // Prepare feature vectors [R, G, B, X, Y]
    std::vector<float> features(total_pixels * 5);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            Vec3b color = image.at<Vec3b>(i, j);
            features[idx * 5 + 0] = color[0];
            features[idx * 5 + 1] = color[1];
            features[idx * 5 + 2] = color[2];
            features[idx * 5 + 3] = j;
            features[idx * 5 + 4] = i;
        }
    }

    // Initialize centroids
    std::vector<float> centroids(k * 5);
    srand(12345);
    for (int i = 0; i < k; ++i) {
        int idx = rand() % total_pixels;
        for (int j = 0; j < 5; ++j)
            centroids[i * 5 + j] = features[idx * 5 + j];
    }

    // K-means iterations
    std::vector<int> assignments(total_pixels);
    bool converged = false;
    int iter = 0;
    
    while (!converged && iter < max_iters) {
        // Assign points to clusters
        #pragma omp parallel for
        for (int i = 0; i < total_pixels; ++i) {
            float min_dist = INFINITY;
            int cluster = 0;
            
            for (int c = 0; c < k; ++c) {
                float dist = 0.0f;
                for (int j = 0; j < 5; ++j) {
                    float diff = features[i * 5 + j] - centroids[c * 5 + j];
                    dist += diff * diff;
                }
                if (dist < min_dist) {
                    min_dist = dist;
                    cluster = c;
                }
            }
            assignments[i] = cluster;
        }

        // Update centroids
        std::vector<float> new_centroids(k * 5, 0.0f);
        std::vector<int> counts(k, 0);
        
        #pragma omp parallel
        {
            std::vector<float> local_centroids(k * 5, 0.0f);
            std::vector<int> local_counts(k, 0);
            
            #pragma omp for nowait
            for (int i = 0; i < total_pixels; ++i) {
                int cluster = assignments[i];
                for (int j = 0; j < 5; ++j)
                    local_centroids[cluster * 5 + j] += features[i * 5 + j];
                local_counts[cluster]++;
            }
            
            #pragma omp critical
            {
                for (int c = 0; c < k; ++c) {
                    counts[c] += local_counts[c];
                    for (int j = 0; j < 5; ++j)
                        new_centroids[c * 5 + j] += local_centroids[c * 5 + j];
                }
            }
        }

        // Normalize and check convergence
        float max_diff = 0.0f;
        for (int c = 0; c < k; ++c) {
            if (counts[c] == 0) continue;
            
            for (int j = 0; j < 5; ++j)
                new_centroids[c * 5 + j] /= counts[c];
            
            float diff = 0.0f;
            for (int j = 0; j < 5; ++j) {
                float d = new_centroids[c * 5 + j] - centroids[c * 5 + j];
                diff += d * d;
            }
            max_diff = std::max(max_diff, std::sqrt(diff));
        }
        
        converged = (max_diff < epsilon);
        if (!converged)
            centroids.swap(new_centroids);
        
        iter++;
    }

    // Apply segmentation
    #pragma omp parallel for
    for (int i = 0; i < total_pixels; ++i) {
        int cluster = assignments[i];
        features[i * 5 + 0] = centroids[cluster * 5 + 0];
        features[i * 5 + 1] = centroids[cluster * 5 + 1];
        features[i * 5 + 2] = centroids[cluster * 5 + 2];
    }

    // Save result
    Mat segmented(rows, cols, CV_8UC3);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            segmented.at<Vec3b>(i, j) = Vec3b(
                static_cast<uchar>(features[idx * 5 + 0]),
                static_cast<uchar>(features[idx * 5 + 1]),
                static_cast<uchar>(features[idx * 5 + 2])
            );
        }
    }
    
    cvtColor(segmented, segmented, COLOR_RGB2BGR);
    imwrite("output_omp.png", segmented);
    std::cout << "Segmentation complete. Saved as output_omp.png" << std::endl;

    return 0;
}