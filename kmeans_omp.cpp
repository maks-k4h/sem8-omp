#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include <omp.h>

using namespace cv;

int main() {
    const int k = 5;         // Number of clusters
    const int max_iters = 100; // Max iterations
    const float epsilon = 1.0f; // Convergence threshold

    // Read input image
    Mat image = imread("input.png");
    if (image.empty()) {
        std::cerr << "Error: Could not read image" << std::endl;
        return 1;
    }
    cvtColor(image, image, COLOR_BGR2RGB);

    // Prepare feature data [R, G, B, X, Y]
    const int rows = image.rows;
    const int cols = image.cols;
    const int total_pixels = rows * cols;
    std::vector<float> features(total_pixels * 5);

    #pragma omp parallel for
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            int idx = y * cols + x;
            Vec3b color = image.at<Vec3b>(y, x);
            features[idx * 5 + 0] = color[0];
            features[idx * 5 + 1] = color[1];
            features[idx * 5 + 2] = color[2];
            features[idx * 5 + 3] = x;
            features[idx * 5 + 4] = y;
        }
    }

    // Initialize centroids
    std::vector<float> centroids(k * 5);
    srand(12345);
    for (int i = 0; i < k; i++) {
        int idx = rand() % total_pixels;
        for (int j = 0; j < 5; j++)
            centroids[i * 5 + j] = features[idx * 5 + j];
    }

    // K-means main loop
    std::vector<float> new_centroids(k * 5);
    bool converged = false;
    int iter = 0;

    while (!converged && iter++ < max_iters) {
        std::vector<float> sum(k * 5, 0.0f);
        std::vector<int> count(k, 0);

        // Parallel cluster assignment
        #pragma omp parallel
        {
            std::vector<float> local_sum(k * 5, 0.0f);
            std::vector<int> local_count(k, 0);

            #pragma omp for
            for (int i = 0; i < total_pixels; i++) {
                float min_dist = INFINITY;
                int cluster = -1;
                const float* pixel = &features[i * 5];

                // Find nearest centroid
                for (int c = 0; c < k; c++) {
                    float dist = 0.0f;
                    for (int j = 0; j < 5; j++) {
                        float diff = pixel[j] - centroids[c * 5 + j];
                        dist += diff * diff;
                    }
                    if (dist < min_dist) {
                        min_dist = dist;
                        cluster = c;
                    }
                }

                // Update local sums
                for (int j = 0; j < 5; j++)
                    local_sum[cluster * 5 + j] += pixel[j];
                local_count[cluster]++;
            }

            // Merge thread-local results
            #pragma omp critical
            {
                for (int c = 0; c < k; c++) {
                    count[c] += local_count[c];
                    for (int j = 0; j < 5; j++)
                        sum[c * 5 + j] += local_sum[c * 5 + j];
                }
            }
        }

        // Update centroids
        float max_diff = 0.0f;
        for (int c = 0; c < k; c++) {
            if (count[c] == 0) continue;
            
            for (int j = 0; j < 5; j++)
                new_centroids[c * 5 + j] = sum[c * 5 + j] / count[c];

            // Calculate convergence
            float dist = 0.0f;
            for (int j = 0; j < 5; j++) {
                float diff = new_centroids[c * 5 + j] - centroids[c * 5 + j];
                dist += diff * diff;
            }
            max_diff = std::max(max_diff, std::sqrt(dist));
        }
        
        centroids.swap(new_centroids);
        converged = (max_diff < epsilon);
    }

    // Apply final colors
    #pragma omp parallel for
    for (int i = 0; i < total_pixels; i++) {
        float* pixel = &features[i * 5];
        float min_dist = INFINITY;
        int cluster = -1;

        for (int c = 0; c < k; c++) {
            float dist = 0.0f;
            for (int j = 0; j < 5; j++) {
                float diff = pixel[j] - centroids[c * 5 + j];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                cluster = c;
            }
        }

        pixel[0] = centroids[cluster * 5 + 0];
        pixel[1] = centroids[cluster * 5 + 1];
        pixel[2] = centroids[cluster * 5 + 2];
    }

    // Save output
    Mat result(rows, cols, CV_8UC3);
    #pragma omp parallel for
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            int idx = y * cols + x;
            result.at<Vec3b>(y, x) = Vec3b(
                static_cast<uchar>(features[idx * 5 + 0]),
                static_cast<uchar>(features[idx * 5 + 1]),
                static_cast<uchar>(features[idx * 5 + 2])
            );
        }
    }
    cvtColor(result, result, COLOR_RGB2BGR);
    imwrite("output.png", result);

    std::cout << "Segmentation complete. Saved as output.png" << std::endl;
    return 0;
}