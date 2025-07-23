import numpy as np
import cv2 as cv


class RegionMemory:
    def __init__(self):
        self.panorama_history = None
        self.homography_history = {}
        
    def should_update_region(self, current_region, stored_region):
        if stored_region is None:
            return True
        # So sánh đặc trưng cơ bản
        diff = np.abs(current_region.astype(np.int16) - stored_region.astype(np.int16))
        significant_diff = np.mean(diff) > 15
        
        return significant_diff
    
    def update_panorama(self, new_panorama, changed_mask=None):
        if self.panorama_history is None:
            self.panorama_history = new_panorama.copy()
            return new_panorama

        if changed_mask is None or np.sum(changed_mask) > 0.2 * changed_mask.size:
            # Nếu có ít nhất 10% ảnh thay đổi, cập nhật toàn bộ
            self.panorama_history = new_panorama.copy()
            return new_panorama

        return self.panorama_history


# Khởi tạo bộ nhớ vùng
region_memory = RegionMemory()
prev_images = {}  # Lưu trữ ảnh trước đó của mỗi camera


def opticalflow_motion_detection(img1, prev_img1, motion_threshold=0.5, min_moving_points_ratio=0.008):
    if prev_img1 is None or img1 is None:
        return True

    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray_prev = cv.cvtColor(prev_img1, cv.COLOR_BGR2GRAY)

    feature_params = dict(maxCorners=800, qualityLevel=0.001, minDistance=4, blockSize=7)
    p0 = cv.goodFeaturesToTrack(gray_prev, mask=None, **feature_params)

    if p0 is None:
        return True

    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    p1, st, err = cv.calcOpticalFlowPyrLK(gray_prev, gray1, p0, None, **lk_params)

    if p1 is None or st is None:
        return True

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    motion_vectors = good_new - good_old
    magnitudes = np.linalg.norm(motion_vectors, axis=1)

    moving_points = np.sum(magnitudes > motion_threshold)
    total_points = len(magnitudes)

    moving_ratio = moving_points / total_points

    return moving_ratio > min_moving_points_ratio



def adaptive_sift(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    variance = np.var(gray)

    if variance < 50:
        # Ảnh phẳng, tăng nhạy
        contrastThreshold = 0.01
        edgeThreshold = 5
        nfeatures = 3000
    elif variance > 300:
        # Ảnh nhiều chi tiết, lọc bớt nhiễu
        contrastThreshold = 0.04
        edgeThreshold = 12
        nfeatures = 2000
    else:
        # Trung bình
        contrastThreshold = 0.03
        edgeThreshold = 10
        nfeatures = 2500

    sift = cv.SIFT_create(nfeatures=nfeatures, contrastThreshold=contrastThreshold, edgeThreshold=edgeThreshold)
    return sift

def sift_detect(image: list):
    keypoints = []
    descriptors = []
    for img in image:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        sift = adaptive_sift(img)
        kp, des = sift.detectAndCompute(gray, None)
        keypoints.append(kp)
        descriptors.append(des)
    return keypoints, descriptors

def adaptive_ratio_thresh(num_matches, for_local_homography=False):
    if for_local_homography:
        # Nới lỏng ngưỡng lọc khi dùng cho homography cục bộ
        if num_matches < 50:
            return 0.9  # Nới lỏng hơn để nhận nhiều điểm hơn
        elif num_matches > 200:
            return 0.75
        else:
            return 0.85
    else:
        # Ngưỡng gốc cho homography toàn cục
        if num_matches < 50:
            return 0.8
        elif num_matches > 200:
            return 0.65
        else:
            return 0.7
        
def matcher(descriptors1, descriptors2, kp1, kp2, for_local_homography=False):
    if descriptors1 is None or descriptors2 is None:
        return []
    if len(descriptors1) < 2 or len(descriptors2) < 2:
        return None

    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)

    good_matches = []
    for match_pair in knn_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < adaptive_ratio_thresh(len(match_pair), for_local_homography) * n.distance:
                good_matches.append(m)
    
    min_matches = 4 if for_local_homography else 6
    if len(good_matches) < min_matches:  # Giảm ngưỡng cho homography cục bộ
        return []
        
    def filter_matches_spatially(kp1, kp2, matches):
        center_img1 = np.mean([kp.pt for kp in kp1], axis=0)
        center_img2 = np.mean([kp.pt for kp in kp2], axis=0)
        filtered = []
        for m in matches:
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt
            d1 = np.linalg.norm(np.array(pt1) - center_img1)
            d2 = np.linalg.norm(np.array(pt2) - center_img2)
            
            # Nới lỏng điều kiện không gian khi dùng cho homography cục bộ
            spatial_thresh = 0.85 if for_local_homography else 0.7
            if d1 < spatial_thresh * max(center_img1) and d2 < spatial_thresh * max(center_img2):
                filtered.append(m)
        return filtered
        
    good_matches = filter_matches_spatially(kp1, kp2, good_matches)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    if len(good_matches) < min_matches:
        return []
        
    # Giảm độ nghiêm ngặt RANSAC cho homography cục bộ
    ransac_thresh = 5.0 if for_local_homography else 4.0
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, ransac_thresh)

    if mask is None:
        return []

    refined_matches = [m for m, inlier in zip(good_matches, mask.ravel()) if inlier]

    return refined_matches


def show_matcher(img1, keypoints1, img2, keypoints2, good_matches):
    img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow('Good Matches', img_matches)
    cv.waitKey()

def dynamic_good_match_threshold(kp1, kp2):
    total_keypoints = len(kp1) + len(kp2)

    if total_keypoints < 1500:
        return 5 
    elif total_keypoints < 4500:
        return 15
    else:
        return 20 
    
class MatchFailureTracker:
    def __init__(self):
        self.failure_count = 0

    def update(self, success):
        if success:
            self.failure_count = 0
        else:
            self.failure_count += 1

    def get_threshold_multiplier(self):
        # Nếu thất bại liên tục, nới lỏng threshold dần
        if self.failure_count >= 3:
            return 0.5
        elif self.failure_count >= 5:
            return 0.3
        return 1.0

failure_tracker = MatchFailureTracker()

class HomographyStabilizer:
    def __init__(self, threshold=3, buffer_size=15):
        self.threshold = threshold  
        self.buffer_size = buffer_size
        self.stable_H = None
        self.H_buffer = []
        self.frame_count = 0
        self.last_valid_H = None

    def is_valid_homography(self, H):
        if H is None:
            return False
        det = np.linalg.det(H[:2, :2])
        if abs(det) < 0.1 or abs(det) > 10:
            return False
        if np.any(np.abs(H) > 1e4) or np.any(np.isnan(H)):
            return False
        return True
        
    def stabilize_homography(self, H_new):
        if not self.is_valid_homography(H_new):
            if self.last_valid_H is not None:
                return self.last_valid_H
            if self.stable_H is not None:
                return self.stable_H
            return np.eye(3, dtype=np.float32)

        # Lưu H hợp lệ cuối cùng
        self.last_valid_H = H_new.copy()
        
        if self.stable_H is None:
            self.stable_H = H_new.copy()
            return self.stable_H

        diff = np.linalg.norm(H_new - self.stable_H)

        if diff < self.threshold:
            return self.stable_H

        if diff > self.threshold * 5:
            if self.is_valid_homography(H_new):
                self.stable_H = H_new.copy()
            return self.stable_H

        alpha = min(diff / (self.threshold * 2), 0.3)
        self.stable_H = (1 - alpha) * self.stable_H + alpha * H_new
        self.frame_count += 1
        return self.stable_H
        
    def reset(self):
        self.stable_H = None
        self.H_buffer = []
        self.frame_count = 0
        self.last_valid_H = None

stabilizer = HomographyStabilizer(threshold=10, buffer_size=15)

def color_correct_images(img1, img2, mask1, mask2):
    # Tạo mask cho vùng chồng lấp
    overlap = cv.bitwise_and(mask1, mask2)
    
    if np.sum(overlap) == 0:
        return img1, img2
    
    # Tạo mask để chỉ tính giá trị trong vùng chồng lấp
    overlap_3ch = cv.cvtColor(overlap, cv.COLOR_GRAY2BGR)
    
    # Tính giá trị trung bình của mỗi kênh màu trong vùng chồng lấp
    mean_img1 = cv.mean(img1, mask=overlap)[:3]
    mean_img2 = cv.mean(img2, mask=overlap)[:3]
    
    # Tính tỷ lệ gain để áp dụng cho img2
    gain = np.array([mean_img1[i] / mean_img2[i] if mean_img2[i] > 0 else 1.0 for i in range(3)])
    gain = np.clip(gain, 0.7, 1.5)  # Giới hạn gain để tránh hiệu ứng quá mức
    
    # Áp dụng hiệu chỉnh màu cho img2
    adjusted_img2 = img2.copy().astype(np.float32)
    adjusted_img2[:, :, 0] *= gain[0]
    adjusted_img2[:, :, 1] *= gain[1]
    adjusted_img2[:, :, 2] *= gain[2]
    
    # Giới hạn giá trị trong khoảng [0, 255]
    adjusted_img2 = np.clip(adjusted_img2, 0, 255).astype(np.uint8)
    
    return img1, adjusted_img2

def multi_band_blend(img1, img2, mask1, mask2):
    # Tạo mask cho vùng chồng lấp
    overlap = cv.bitwise_and(mask1, mask2)
    
    # Nếu không có vùng chồng lấp, sử dụng phương pháp đơn giản
    if np.sum(overlap) == 0:
        result = np.zeros_like(img1)
        result = np.where(mask1[:, :, None] > 0, img1, result)
        result = np.where(mask2[:, :, None] > 0, img2, result)
        return result
    
    # Chuyển đổi mask sang định dạng float 32-bit
    mask1_f = mask1.astype(np.float32) / 255.0
    mask2_f = mask2.astype(np.float32) / 255.0
    
    # Làm mịn mask bằng Gaussian blur
    h, w = mask1.shape
    kernel_size = max(5, w // 15)
    mask1_blurred = cv.GaussianBlur(mask1_f, (kernel_size | 1, kernel_size | 1), 0)
    mask2_blurred = cv.GaussianBlur(mask2_f, (kernel_size | 1, kernel_size | 1), 0)
    # Tạo weight cho mỗi ảnh
    weight1 = mask1_blurred / (mask1_blurred + mask2_blurred + 1e-10)
    weight2 = 1.0 - weight1
    
    # Áp dụng weight cho mỗi ảnh
    img1_f = img1.astype(np.float32)
    img2_f = img2.astype(np.float32)
    
    # Tạo kết quả bằng cách trộn ảnh dựa trên weight
    result = np.zeros_like(img1_f)
    for c in range(3):
        result[:, :, c] = img1_f[:, :, c] * weight1 + img2_f[:, :, c] * weight2
    
    return result.astype(np.uint8)
def find_optimal_seam(img1, img2, mask1, mask2):
    # Tạo mask cho vùng chồng lấp
    overlap = cv.bitwise_and(mask1, mask2)
    
    if np.sum(overlap) == 0:
        return mask1, mask2
        
    # Tính gradient magnitude cho từng ảnh
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    
    grad_x1 = cv.Sobel(gray1, cv.CV_32F, 1, 0)
    grad_y1 = cv.Sobel(gray1, cv.CV_32F, 0, 1)
    grad_mag1 = cv.magnitude(grad_x1, grad_y1)
    
    grad_x2 = cv.Sobel(gray2, cv.CV_32F, 1, 0)
    grad_y2 = cv.Sobel(gray2, cv.CV_32F, 0, 1)
    grad_mag2 = cv.magnitude(grad_x2, grad_y2)
    
    # Sử dụng gradient magnitude để tìm đường ghép tốt nhất
    grad_diff = np.abs(grad_mag1 - grad_mag2) * overlap.astype(np.float32)
    
    # Chuyển đổi sang uint8 trước khi áp dụng THRESH_OTSU
    grad_diff_uint8 = cv.normalize(grad_diff, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    _, thresh = cv.threshold(grad_diff_uint8, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    dist_transform = cv.distanceTransform(thresh, cv.DIST_L2, 3)
    
    # Tạo mask mới dựa trên distance transform
    norm_dist = cv.normalize(dist_transform, None, 0, 1.0, cv.NORM_MINMAX)
    _, new_mask = cv.threshold(norm_dist, 0.5, 1.0, cv.THRESH_BINARY)
    
    # Cập nhật mask
    new_mask1 = mask1.copy()
    new_mask2 = mask2.copy()
    
    # Cập nhật vùng chồng lấp
    overlap_region = (new_mask * 255).astype(np.uint8) & overlap
    new_mask1[overlap > 0] = overlap_region[overlap > 0]
    new_mask2[overlap > 0] = (255 - overlap_region)[overlap > 0]
    
    return new_mask1, new_mask2

def create_grid(image, grid_size_x=4, grid_size_y=4):
    h, w = image.shape[:2]
    grid_points = []
    for y in range(0, grid_size_y + 1):
        for x in range(0, grid_size_x + 1):
            grid_points.append([x * w / grid_size_x, y * h / grid_size_y])
    return np.array(grid_points, dtype=np.float32).reshape(-1, 1, 2)

def compute_local_homographies(kp1, kp2, good_matches, img_shape, grid_size=4):
    h, w = img_shape[:2]
    grid_h, grid_w = h//grid_size, w//grid_size
    local_H = {}
    
    # Lọc matches vào các ô lưới
    for i in range(grid_size):
        for j in range(grid_size):
            cell_matches = []
            for m in good_matches:
                x, y = kp1[m.queryIdx].pt
                if j*grid_w <= x < (j+1)*grid_w and i*grid_h <= y < (i+1)*grid_h:
                    cell_matches.append(m)
            
            # Nếu có đủ matches, tính homography cho ô lưới này
            if len(cell_matches) >= 4:  # Tối thiểu 4 điểm cho homography
                src_pts = np.float32([kp1[m.queryIdx].pt for m in cell_matches]).reshape(-1,1,2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in cell_matches]).reshape(-1,1,2)
                H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)  # Nới lỏng ngưỡng RANSAC
                if H is not None:
                    local_H[(i, j)] = H
    
    return local_H

def interpolate_homography(point, local_H, grid_size, img_shape):
    h, w = img_shape[:2]
    grid_h, grid_w = h/grid_size, w/grid_size
    
    # Xác định ô lưới chứa điểm - Sửa lại cách truy cập
    x = point[0, 0, 0]  # Truy cập tọa độ x
    y = point[0, 0, 1]  # Truy cập tọa độ y
    
    grid_j = int(x / grid_w)
    grid_i = int(y / grid_h)
    
    # Giới hạn chỉ số trong phạm vi lưới
    grid_i = max(0, min(grid_i, grid_size-1))
    grid_j = max(0, min(grid_j, grid_size-1))
    
    # Lấy homography của ô lưới hoặc sử dụng homography mặc định
    H = local_H.get((grid_i, grid_j))
    
    # Nếu không có H cho ô này, tìm H gần nhất
    if H is None:
        min_dist = float('inf')
        nearest_H = None
        for (i, j), h_matrix in local_H.items():
            dist = abs(i - grid_i) + abs(j - grid_j)
            if dist < min_dist:
                min_dist = dist
                nearest_H = h_matrix
        H = nearest_H if nearest_H is not None else np.eye(3)
    
    return cv.perspectiveTransform(point, H)

def mesh_warp_image(img1, local_H, result_size, grid_size=3):
    h1, w1 = img1.shape[:2]
    result_h, result_w = result_size
    
    # Tạo lưới nguồn và đích
    src_mesh = create_grid(img1, grid_size, grid_size)
    dst_mesh = np.zeros_like(src_mesh)
    
    # Biến đổi từng điểm lưới
    for i in range(len(src_mesh)):
        pt = src_mesh[i].reshape(1, 1, 2)
        dst_mesh[i] = interpolate_homography(pt, local_H, grid_size, img1.shape)
    
    # Tạo ảnh kết quả
    result = np.zeros((result_h, result_w, 3), dtype=np.uint8)
    
    # Chia lưới thành tam giác để warping
    triangles = []
    for i in range(grid_size):
        for j in range(grid_size):
            # Mỗi ô lưới tạo 2 tam giác
            idx1 = i * (grid_size + 1) + j
            idx2 = i * (grid_size + 1) + (j + 1)
            idx3 = (i + 1) * (grid_size + 1) + j
            idx4 = (i + 1) * (grid_size + 1) + (j + 1)
            
            triangles.append([idx1, idx2, idx3])
            triangles.append([idx2, idx4, idx3])
    
    # Áp dụng warping cho từng tam giác
    for tri in triangles:
        try:
            src_tri = np.float32([src_mesh[tri[0]][0], src_mesh[tri[1]][0], src_mesh[tri[2]][0]])
            dst_tri = np.float32([dst_mesh[tri[0]][0], dst_mesh[tri[1]][0], dst_mesh[tri[2]][0]])
            
            # Kiểm tra xem tam giác có hợp lệ không
            if np.any(np.isnan(dst_tri)) or np.any(np.isinf(dst_tri)):
                continue
            
            # Tính ma trận affine cho tam giác
            warp_mat = cv.getAffineTransform(src_tri, dst_tri)
            
            # Tìm bounding box của tam giác đích
            x_min = int(min(dst_tri[:, 0]))
            y_min = int(min(dst_tri[:, 1]))
            x_max = int(max(dst_tri[:, 0]))
            y_max = int(max(dst_tri[:, 1]))
            
            # Kiểm tra xem bounding box có hợp lệ không
            if x_min >= x_max or y_min >= y_max:
                continue
            
            # Giới hạn trong kích thước ảnh kết quả
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(result_w - 1, x_max)
            y_max = min(result_h - 1, y_max)
            
            # Kiểm tra lại sau khi giới hạn
            if x_min >= x_max or y_min >= y_max:
                continue
            
            # Tạo mask cho tam giác
            mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
            tri_points = dst_tri - np.float32([x_min, y_min])
            cv.fillConvexPoly(mask, np.int32(tri_points), 1)
            
            # Áp dụng warp affine
            warped = cv.warpAffine(img1, warp_mat, (result_w, result_h), flags=cv.INTER_LINEAR)
            
            # Cập nhật kết quả
            result[y_min:y_max+1, x_min:x_max+1] = np.where(
                mask[:, :, np.newaxis] > 0,
                warped[y_min:y_max+1, x_min:x_max+1],
                result[y_min:y_max+1, x_min:x_max+1]
            )
        except Exception as e:
            # Bỏ qua tam giác có vấn đề
            continue
    
    return result

def combine_image(good_matches, kp1, kp2, img1, img2):
    threshold = dynamic_good_match_threshold(kp1, kp2)
    multiplier = failure_tracker.get_threshold_multiplier()
    threshold = int(threshold * multiplier)
    if len(good_matches) < threshold:
        failure_tracker.update(False)
        print(f"Not enough matches ({len(good_matches)}) vs threshold ({threshold})")
        return None
    else:
        failure_tracker.update(True)
    if len(good_matches) < 4:
        failure_tracker.update(False)
        print(f"Not enough matches ({len(good_matches)}) to compute Homography")
        return None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Tính homography toàn cục cho việc ước tính kích thước và homography cục bộ
    H_ransac, mask_ransac = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 3.0)
    
    # Tính homography cục bộ
    grid_size = 3  # Số lưới theo mỗi chiều
    
    # Tạo matches mới với tiêu chí nới lỏng hơn cho homography cục bộ
    # Chỉ áp dụng khi có đủ số lượng good_matches ban đầu để đảm bảo có đủ keypoints
    if len(good_matches) >= 20:
        local_matches = matcher(
            cv.SIFT_create().compute(cv.cvtColor(img1, cv.COLOR_BGR2GRAY), kp1)[1], 
            cv.SIFT_create().compute(cv.cvtColor(img2, cv.COLOR_BGR2GRAY), kp2)[1],
            kp1, kp2, for_local_homography=True
        )
        if len(local_matches) > len(good_matches):
            print(f"Sử dụng {len(local_matches)} điểm trùng khớp nới lỏng cho homography cục bộ (ban đầu {len(good_matches)})")
            local_H = compute_local_homographies(kp1, kp2, local_matches, img1.shape, grid_size)
        else:
            local_H = compute_local_homographies(kp1, kp2, good_matches, img1.shape, grid_size)
    else:
        local_H = compute_local_homographies(kp1, kp2, good_matches, img1.shape, grid_size)
    
    # Kiểm tra nếu không đủ homography cục bộ, sử dụng homography toàn cục
    if len(local_H) < grid_size * grid_size / 2:
        print(f"Không đủ homography cục bộ ({len(local_H)}), sử dụng homography toàn cục")
        H = H_ransac if H_ransac is not None else np.eye(3, dtype=np.float32)
        H = stabilizer.stabilize_homography(H)
        
        # Sử dụng phương pháp ghép ảnh cũ
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        corners_img1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
        warped_corners = cv.perspectiveTransform(corners_img1, H)
        
        corners_img2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        
        all_corners = np.concatenate([warped_corners, corners_img2], axis=0)
        
        x_min = int(np.min(all_corners[:, 0, 0]))
        x_max = int(np.max(all_corners[:, 0, 0]))
        y_min = int(np.min(all_corners[:, 0, 1]))
        y_max = int(np.max(all_corners[:, 0, 1]))
        
        translation_matrix = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ], dtype=np.float32)
        
        H_combined = translation_matrix @ H
        
        result_width = (x_max - x_min)
        result_height = (y_max - y_min)
        
        MAX_CANVAS_WIDTH = 4500
        MAX_CANVAS_HEIGHT = 4500
        if result_width > MAX_CANVAS_WIDTH or result_height > MAX_CANVAS_HEIGHT:
            return None
            
        result = np.zeros((result_height, result_width, 3), dtype=np.uint8)
        
        warped_img1 = cv.warpPerspective(img1, H_combined, (result_width, result_height))
        
        translation_img2 = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ], dtype=np.float32)
        
        warped_img2 = cv.warpPerspective(img2, translation_img2, (result_width, result_height))
    else:
        # Sử dụng homography cục bộ để ghép ảnh
        print(f"Sử dụng {len(local_H)} homography cục bộ cho mesh warping")
        
        # Tính kích thước ảnh kết quả dựa vào homography toàn cục
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        corners_img1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
        
        if H_ransac is not None:
            warped_corners = cv.perspectiveTransform(corners_img1, H_ransac)
            corners_img2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
            all_corners = np.concatenate([warped_corners, corners_img2], axis=0)
            
            x_min = int(np.min(all_corners[:, 0, 0]))
            x_max = int(np.max(all_corners[:, 0, 0]))
            y_min = int(np.min(all_corners[:, 0, 1]))
            y_max = int(np.max(all_corners[:, 0, 1]))
            
            result_width = (x_max - x_min)
            result_height = (y_max - y_min)
            
            # Điều chỉnh local_H để tính đến dịch chuyển
            for key in local_H:
                translation = np.array([
                    [1, 0, -x_min],
                    [0, 1, -y_min],
                    [0, 0, 1]
                ], dtype=np.float32)
                local_H[key] = translation @ local_H[key]
            
            # Điều chỉnh translation cho ảnh thứ hai
            translation_img2 = np.array([
                [1, 0, -x_min],
                [0, 1, -y_min],
                [0, 0, 1]
            ], dtype=np.float32)
        else:
            # Nếu không có homography toàn cục, ước tính kích thước ảnh kết quả
            result_width = w1 + w2
            result_height = max(h1, h2)
            x_min = 0
            y_min = 0
            
            translation_img2 = np.array([
                [1, 0, w1],
                [0, 1, 0],
                [0, 0, 1]
            ], dtype=np.float32)
        
        # Kiểm tra kích thước tối đa
        MAX_CANVAS_WIDTH = 4500
        MAX_CANVAS_HEIGHT = 4500
        if result_width > MAX_CANVAS_WIDTH or result_height > MAX_CANVAS_HEIGHT:
            return None
            
        # Áp dụng mesh warping cho ảnh thứ nhất
        result_size = (result_height, result_width)
        warped_img1 = mesh_warp_image(img1, local_H, result_size, grid_size)
        
        # Warp ảnh thứ hai
        warped_img2 = cv.warpPerspective(img2, translation_img2, (result_width, result_height))
    
    # Tạo mask cho từng ảnh
    mask1 = np.any(warped_img1 > 0, axis=2).astype(np.uint8) * 255
    mask2 = np.any(warped_img2 > 0, axis=2).astype(np.uint8) * 255
    
    # Tìm đường ghép tối ưu
    seam_mask1, seam_mask2 = find_optimal_seam(warped_img1, warped_img2, mask1, mask2)
    
    # Hiệu chỉnh màu sắc
    color_warped_img1, color_warped_img2 = color_correct_images(warped_img1, warped_img2, seam_mask1, seam_mask2)
    
    # Multi-band blending
    result = multi_band_blend(color_warped_img1, color_warped_img2, seam_mask1, seam_mask2)
    
    # Phát hiện vùng thay đổi
    changed_regions = np.zeros((result.shape[0], result.shape[1]), dtype=np.uint8)
    
    # Kiểm tra thay đổi với ảnh panorama trước đó
    if region_memory.panorama_history is not None:
        old_h, old_w = region_memory.panorama_history.shape[:2]
        
        # Nếu kích thước khác nhau, xem là thay đổi toàn bộ
        if old_h != result.shape[0] or old_w != result.shape[1]:
            changed_regions = np.ones((result.shape[0], result.shape[1]), dtype=np.uint8) * 255
        else:
            # Chia thành 4x4 vùng để kiểm tra thay đổi
            grid_h, grid_w = result.shape[0] // 4, result.shape[1] // 4
            
            for i in range(4):
                for j in range(4):
                    y_start = i * grid_h
                    y_end = (i + 1) * grid_h if i < 3 else result.shape[0]
                    x_start = j * grid_w
                    x_end = (j + 1) * grid_w if j < 3 else result.shape[1]
                    
                    if y_end <= y_start or x_end <= x_start:
                        continue
                        
                    region = result[y_start:y_end, x_start:x_end]
                    prev_region = region_memory.panorama_history[y_start:y_end, x_start:x_end]
                    
                    if region_memory.should_update_region(region, prev_region):
                        changed_regions[y_start:y_end, x_start:x_end] = 255
    else:
        # Nếu chưa có panorama trước đó, xem là thay đổi toàn bộ
        changed_regions = np.ones((result.shape[0], result.shape[1]), dtype=np.uint8) * 255
    
    # Cập nhật panorama trong bộ nhớ vùng
    final_result = region_memory.update_panorama(result, changed_regions)
    
    return final_result

def image_stitching(image, ShowMatcher = False):
    global region_memory, prev_images
    
    # Kiểm tra chuyển động
    needs_update = False
    for idx, img in enumerate(image):
        img_key = f"camera_{idx}"
        if img_key in prev_images:
            # Chỉ kiểm tra thay đổi với ảnh đã giảm kích thước
            h, w = img.shape[:2]
            scale = min(1.0, 100 / max(h, w))  # Giảm xuống 100px để tăng tốc
            small_img = cv.resize(img, (int(w * scale), int(h * scale)))
            small_prev = cv.resize(prev_images[img_key], (int(w * scale), int(h * scale)))
            
            if opticalflow_motion_detection(small_img, small_prev):
                needs_update = True

        else:
            needs_update = True
        prev_images[img_key] = img.copy()
    
    # Nếu không có chuyển động đáng kể và đã có kết quả trước đó
    if not needs_update and region_memory.panorama_history is not None:
        return region_memory.panorama_history
    
    # Tiếp tục với thuật toán ghép ảnh
    kp, des = sift_detect(image)
    good_matcher = matcher(des[0], des[1], kp[0], kp[1])
    
    if not good_matcher:
        print("Không tìm thấy điểm trùng khớp")
        if region_memory.panorama_history is not None:
            return region_memory.panorama_history
        return None

    print(f"Tìm thấy {len(good_matcher)} điểm trùng khớp")

    if ShowMatcher:
        show_matcher(image[0], kp[0], image[1], kp[1], good_matcher)

    result = combine_image(good_matcher, kp[0], kp[1], image[0], image[1])

    if result is None:
        print("Ghép ảnh thất bại")
        if region_memory.panorama_history is not None:
            return region_memory.panorama_history
    return result

if __name__ == "__main__":
    img1 = cv.imread('DSC02930.JPG')
    img2 = cv.imread('DSC02931.JPG')
    
    if img1 is None or img2 is None:
        print("Cannot read images")
        exit()
    max_dimension = 1200
    if img1.shape[1] > max_dimension:
        scale = max_dimension / img1.shape[1]
        img1 = cv.resize(img1, None, fx=scale, fy=scale)
        img2 = cv.resize(img2, None, fx=scale, fy=scale)
    
    result = image_stitching([img1, img2])
    cv.imshow('Stitched Image', result)
    cv.waitKey(0)
    cv.destroyAllWindows()
