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


def opticalflow_motion_detection(img1, prev_img1, motion_threshold=0.5, min_moving_points_ratio=0.05):
    if prev_img1 is None or img1 is None:
        return True

    # Giảm kích thước để tính toán nhanh hơn
    h, w = img1.shape[:2]
    max_dim = 100  # Tăng lên từ 50px để có độ chính xác tốt hơn
    
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        small_img1 = cv.resize(img1, None, fx=scale, fy=scale)
        small_prev = cv.resize(prev_img1, None, fx=scale, fy=scale)
    else:
        small_img1 = img1
        small_prev = prev_img1
    
    # Chuyển sang ảnh xám
    gray1 = cv.cvtColor(small_img1, cv.COLOR_BGR2GRAY)
    gray_prev = cv.cvtColor(small_prev, cv.COLOR_BGR2GRAY)
    
    # Sử dụng optical flow nhưng với số lượng điểm giới hạn
    feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=5, blockSize=5)
    p0 = cv.goodFeaturesToTrack(gray_prev, mask=None, **feature_params)
    
    if p0 is None or len(p0) < 10:
        # Không đủ điểm đặc trưng, sử dụng phương pháp so sánh trực tiếp
        diff = cv.absdiff(small_img1, small_prev)
        gray_diff = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        _, thresh_diff = cv.threshold(gray_diff, 25, 255, cv.THRESH_BINARY)
        motion_ratio = np.sum(thresh_diff) / thresh_diff.size
        return motion_ratio > 0.05  # Ngưỡng nhỏ hơn để phát hiện chuyển động nhỏ
    
    # Tối ưu tham số optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    
    p1, st, err = cv.calcOpticalFlowPyrLK(gray_prev, gray1, p0, None, **lk_params)
    
    if p1 is None or st is None:
        return True
    
    # Lấy các điểm tốt
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    
    if len(good_new) == 0:
        return True
    
    # Tính vector chuyển động
    motion_vectors = good_new - good_old
    magnitudes = np.linalg.norm(motion_vectors, axis=1)
    
    # Đếm số điểm chuyển động
    moving_points = np.sum(magnitudes > motion_threshold)
    total_points = len(magnitudes)
    
    if total_points == 0:
        return True
    
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

def adaptive_ratio_thresh(num_matches):
    if num_matches < 50:
        return 0.8
    elif num_matches > 200:
        return 0.65
    else:
        return 0.7
        
def matcher(descriptors1, descriptors2, kp1, kp2):
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
            if m.distance < adaptive_ratio_thresh(len(match_pair)) * n.distance:
                good_matches.append(m)
    
    min_matches = 6
    if len(good_matches) < min_matches:
        return []
        
    # Lọc matches theo không gian
    good_matches = filter_matches_spatially(kp1, kp2, good_matches)
    
    return good_matches

def filter_matches_spatially(kp1, kp2, matches):
    center_img1 = np.mean([kp.pt for kp in kp1], axis=0)
    center_img2 = np.mean([kp.pt for kp in kp2], axis=0)
    filtered = []
    for m in matches:
        pt1 = kp1[m.queryIdx].pt
        pt2 = kp2[m.trainIdx].pt
        d1 = np.linalg.norm(np.array(pt1) - center_img1)
        d2 = np.linalg.norm(np.array(pt2) - center_img2)
        
        spatial_thresh = 0.7
        if d1 < spatial_thresh * max(center_img1) and d2 < spatial_thresh * max(center_img2):
            filtered.append(m)
    return filtered

def show_matcher(img1, keypoints1, img2, keypoints2, good_matches):
    img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow('Good Matches', img_matches)
    cv.waitKey()

def dynamic_good_match_threshold(kp1, kp2):
    total_keypoints = len(kp1) + len(kp2)

    # Giảm ngưỡng để chấp nhận nhiều trường hợp hơn
    if total_keypoints < 1000:
        return 4  # Giảm từ 5 xuống 4 
    elif total_keypoints < 4000:
        return 10  # Giảm từ 15 xuống 10
    else:
        return 15  # Giảm từ 20 xuống 15
    
class MatchFailureTracker:
    def __init__(self):
        self.failure_count = 0

    def update(self, success):
        if success:
            self.failure_count = 0
        else:
            self.failure_count += 1

    def get_threshold_multiplier(self):
        # Nếu thất bại liên tục, nới lỏng threshold dần mạnh hơn
        if self.failure_count >= 2:  # Giảm từ 3 xuống 2
            return 0.4  # Nới lỏng hơn: 0.5 xuống 0.4
        elif self.failure_count >= 4:  # Giảm từ 5 xuống 4
            return 0.3
        return 1.0

failure_tracker = MatchFailureTracker()

def fast_crop_black_borders(img):
    """Cắt bỏ viền đen thừa nhanh hơn bằng quét toàn bộ ảnh"""
    if img is None:
        return None
    
    # Chuyển đổi sang ảnh xám
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Nhị phân hóa với ngưỡng cao hơn để phát hiện tốt hơn
    _, thresh = cv.threshold(gray, 20, 255, cv.THRESH_BINARY)
    
    # Áp dụng phép toán hình thái học để loại bỏ nhiễu và liên kết các vùng
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    
    # Quét toàn bộ ảnh thay vì chỉ một phần
    # Quét từ trái sang phải
    x_left = 0
    for x in range(0, w):
        if np.any(thresh[:, x] > 0):
            x_left = x
            break
    
    # Quét từ phải sang trái
    x_right = w-1
    for x in range(w-1, -1, -1):
        if np.any(thresh[:, x] > 0):
            x_right = x
            break
    
    # Quét từ trên xuống dưới
    y_top = 0
    for y in range(0, h):
        if np.any(thresh[y, :] > 0):
            y_top = y
            break
    
    # Quét từ dưới lên trên
    y_bottom = h-1
    for y in range(h-1, -1, -1):
        if np.any(thresh[y, :] > 0):
            y_bottom = y
            break
    
    # Kiểm tra xem có thực sự tìm thấy vùng có nội dung không
    if x_left >= x_right or y_top >= y_bottom:
        # Không tìm thấy vùng nội dung hợp lệ, trả về ảnh gốc
        return img
    
    # Thêm padding
    padding = 5  # Tăng padding lên để không cắt quá sát
    x_left = max(0, x_left - padding)
    y_top = max(0, y_top - padding)
    x_right = min(w-1, x_right + padding)
    y_bottom = min(h-1, y_bottom + padding)
    
    # Kiểm tra kích thước cắt để tránh cắt quá nhiều
    if (x_right - x_left) < w * 0.3 or (y_bottom - y_top) < h * 0.3:
        # Nếu vùng cắt quá nhỏ (ít hơn 30% kích thước gốc), giữ lại ảnh gốc
        return img
    
    # Cắt ảnh
    return img[y_top:y_bottom+1, x_left:x_right+1]

def simple_blend(img1, img2, mask1, mask2):
    """Trộn hai ảnh với alpha blending đơn giản"""
    # Tạo mask chồng lấp
    overlap = cv.bitwise_and(mask1, mask2)
    
    if np.sum(overlap) == 0:
        # Không có chồng lấp - ghép trực tiếp
        result = np.zeros_like(img1)
        result = np.where(np.repeat(mask1[:, :, np.newaxis], 3, axis=2) > 0, img1, result)
        result = np.where(np.repeat(mask2[:, :, np.newaxis], 3, axis=2) > 0, img2, result)
        return result
    
    # Làm mịn vùng chồng lấp để tạo chuyển tiếp mượt
    kernel_size = 9  # Kernel cho làm mịn (7x7 hoặc 9x9 như đề xuất)
    overlap_float = overlap.astype(np.float32) / 255.0
    overlap_blur = cv.GaussianBlur(overlap_float, (kernel_size, kernel_size), 0)
    
    # Tạo mask 3 kênh
    mask1_3ch = np.repeat(mask1[:, :, np.newaxis] / 255.0, 3, axis=2)
    mask2_3ch = np.repeat(mask2[:, :, np.newaxis] / 255.0, 3, axis=2)
    overlap_3ch = np.repeat(overlap_blur[:, :, np.newaxis], 3, axis=2)
    
    # Tính alpha cho từng pixel - phiên bản đơn giản hơn
    weight = 0.5  # Trọng số cố định cho hai ảnh trong vùng chồng lấp
    
    # Trộn ảnh
    result = np.zeros_like(img1, dtype=np.float32)
    
    # Vùng không chồng lấp của ảnh 1
    mask1_only = (mask1 > 0) & (overlap == 0)
    result[mask1_only] = img1[mask1_only]
    
    # Vùng không chồng lấp của ảnh 2
    mask2_only = (mask2 > 0) & (overlap == 0)
    result[mask2_only] = img2[mask2_only]
    
    # Vùng chồng lấp - alpha blending
    overlap_mask = overlap > 0
    result[overlap_mask] = (img1[overlap_mask] * (1-weight) + 
                          img2[overlap_mask] * weight)
    
    return result.astype(np.uint8)

def multi_band_blend(img1, img2, mask1, mask2):
    """Trộn ảnh đa dải cải tiến với làm mịn vùng chuyển tiếp tốt hơn"""
    # Tạo mask cho vùng chồng lấp
    overlap = cv.bitwise_and(mask1, mask2)
    
    # Nếu không có vùng chồng lấp, sử dụng phương pháp đơn giản
    if np.sum(overlap) == 0:
        result = np.zeros_like(img1)
        result = np.where(mask1[:, :, None] > 0, img1, result)
        result = np.where(mask2[:, :, None] > 0, img2, result)
        return result
    
    # Mở rộng vùng chuyển tiếp để blend mượt hơn
    kernel_size = 31  # Tăng kích thước kernel
    dilate_kernel = np.ones((kernel_size, kernel_size), np.uint8)
    overlap_dilated = cv.dilate(overlap, dilate_kernel)
    
    # Cập nhật mask dựa trên vùng chuyển tiếp mở rộng
    mask1_expanded = mask1.copy()
    mask2_expanded = mask2.copy()
    transition_zone = cv.bitwise_and(overlap_dilated, cv.bitwise_not(overlap))
    
    # Làm mịn vùng chuyển tiếp
    transition_mask = cv.GaussianBlur(transition_zone.astype(np.float32), (kernel_size, kernel_size), 0)
    transition_mask = cv.normalize(transition_mask, None, 0, 1, cv.NORM_MINMAX)
    
    # Xác định kích thước ảnh ghép để tính số lớp pyramid phù hợp
    h, w = mask1.shape
    max_dimension = max(h, w)
    # Số lớp pyramid thích ứng với kích thước ảnh
    num_levels = int(np.log2(max_dimension)) - 2
    num_levels = max(2, min(num_levels, 6))  # Giới hạn từ 2-6 lớp
    
    print(f"Sử dụng {num_levels} lớp pyramid cho multi-band blending")
    
    # Chuyển đổi mask sang định dạng float 32-bit
    mask1_f = mask1.astype(np.float32) / 255.0
    mask2_f = mask2.astype(np.float32) / 255.0
    
    # Tạo Gaussian pyramid cho masks
    mask1_pyr = [mask1_f]
    mask2_pyr = [mask2_f]
    
    for i in range(num_levels - 1):
        mask1_pyr.append(cv.pyrDown(mask1_pyr[-1]))
        mask2_pyr.append(cv.pyrDown(mask2_pyr[-1]))
    
    # Điều chỉnh weight mask để có chuyển tiếp mượt hơn
    for i in range(len(mask1_pyr)):
        # Áp dụng blur hợp lý cho từng lớp - tăng kích thước kernel theo cấp
        kernel_size = 5 + i*4  # Tăng dần theo mức pyramid
        
        # Làm mịn mask
        mask1_pyr[i] = cv.GaussianBlur(mask1_pyr[i], (kernel_size, kernel_size), 0)
        mask2_pyr[i] = cv.GaussianBlur(mask2_pyr[i], (kernel_size, kernel_size), 0)
        
        # Chuẩn hóa weights
        sum_masks = mask1_pyr[i] + mask2_pyr[i]
        mask1_pyr[i] = np.divide(mask1_pyr[i], sum_masks, out=np.zeros_like(mask1_pyr[i]), where=sum_masks!=0)
        mask2_pyr[i] = np.divide(mask2_pyr[i], sum_masks, out=np.zeros_like(mask2_pyr[i]), where=sum_masks!=0)
    
    # Tạo Gaussian pyramid cho ảnh gốc
    img1_pyr = [img1.astype(np.float32)]
    img2_pyr = [img2.astype(np.float32)]
    
    for i in range(num_levels - 1):
        img1_pyr.append(cv.pyrDown(img1_pyr[-1]))
        img2_pyr.append(cv.pyrDown(img2_pyr[-1]))
    
    # Tạo Laplacian pyramid cho ảnh gốc
    laplacian1 = []
    laplacian2 = []
    
    for i in range(num_levels - 1):
        next_img1 = cv.pyrUp(img1_pyr[i + 1], dstsize=(img1_pyr[i].shape[1], img1_pyr[i].shape[0]))
        laplacian1.append(img1_pyr[i] - next_img1)
        
        next_img2 = cv.pyrUp(img2_pyr[i + 1], dstsize=(img2_pyr[i].shape[1], img2_pyr[i].shape[0]))
        laplacian2.append(img2_pyr[i] - next_img2)
    
    # Thêm lớp cuối cùng
    laplacian1.append(img1_pyr[-1])
    laplacian2.append(img2_pyr[-1])
    
    # Kết hợp Laplacian pyramids sử dụng masks
    blended_pyr = []
    
    for i in range(num_levels):
        blended_i = np.zeros_like(laplacian1[i])
        for c in range(3):
            blended_i[:, :, c] = laplacian1[i][:, :, c] * mask1_pyr[i] + laplacian2[i][:, :, c] * mask2_pyr[i]
        blended_pyr.append(blended_i)
    
    # Tái tạo ảnh kết quả từ blended pyramid
    blended_img = blended_pyr[-1]
    for i in range(num_levels - 2, -1, -1):
        blended_img = cv.pyrUp(blended_img, dstsize=(blended_pyr[i].shape[1], blended_pyr[i].shape[0]))
        blended_img += blended_pyr[i]
    
    # Giới hạn giá trị pixel
    return np.clip(blended_img, 0, 255).astype(np.uint8)

def find_optimal_seam(img1, img2, mask1, mask2):
    """Tìm đường ghép tối ưu giữa hai ảnh - phiên bản tối ưu tốc độ"""
    # Kiểm tra đầu vào để tránh lỗi NoneType
    if img1 is None or img2 is None or mask1 is None or mask2 is None:
        print("Warning: Input images or masks are None in find_optimal_seam")
        # Trả về mask gốc nếu không thể xử lý
        if mask1 is not None and mask2 is not None:
            return mask1, mask2
        elif mask1 is not None:
            return mask1, np.zeros_like(mask1)
        elif mask2 is not None:
            return np.zeros_like(mask2), mask2
        else:
            # Tạo mask mặc định nếu cả hai đều None
            h, w = img1.shape[:2] if img1 is not None else img2.shape[:2]
            default_mask = np.zeros((h, w), dtype=np.uint8)
            return default_mask, default_mask
    
    # Tạo mask cho vùng chồng lấp
    overlap = cv.bitwise_and(mask1, mask2)
    
    if np.sum(overlap) == 0:
        return mask1, mask2
    
    # Giảm kích thước để tăng tốc độ
    h, w = overlap.shape
    max_dim = 400
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        small_img1 = cv.resize(img1, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        small_img2 = cv.resize(img2, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        small_mask1 = cv.resize(mask1, None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)
        small_mask2 = cv.resize(mask2, None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)
        small_overlap = cv.resize(overlap, None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)
    else:
        small_img1, small_img2 = img1, img2
        small_mask1, small_mask2 = mask1, mask2
        small_overlap = overlap
    
    # Tạo ảnh gradient
    gray1 = cv.cvtColor(small_img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(small_img2, cv.COLOR_BGR2GRAY)
    
    # Tính gradient với Sobel
    grad_x1 = cv.Sobel(gray1, cv.CV_32F, 1, 0, ksize=3)
    grad_y1 = cv.Sobel(gray1, cv.CV_32F, 0, 1, ksize=3)
    grad_mag1 = cv.magnitude(grad_x1, grad_y1)
    
    grad_x2 = cv.Sobel(gray2, cv.CV_32F, 1, 0, ksize=3)
    grad_y2 = cv.Sobel(gray2, cv.CV_32F, 0, 1, ksize=3)
    grad_mag2 = cv.magnitude(grad_x2, grad_y2)
    
    # Phát hiện đường thẳng để bảo toàn - Giới hạn số lượng đường
    edges1 = cv.Canny(gray1, 50, 150)
    edges2 = cv.Canny(gray2, 50, 150)
    
    # Tạo bản đồ đường thẳng
    line_map1 = np.zeros_like(gray1, dtype=np.float32)
    line_map2 = np.zeros_like(gray2, dtype=np.float32)
    
    # Phát hiện đường thẳng với ít tham số hơn
    lines1 = cv.HoughLinesP(edges1, 1, np.pi/180, threshold=50, 
                         minLineLength=30, maxLineGap=10)
    lines2 = cv.HoughLinesP(edges2, 1, np.pi/180, threshold=50, 
                         minLineLength=30, maxLineGap=10)
    
    # Giới hạn số lượng đường để tăng tốc
    max_lines = 8
    
    # Vẽ đường thẳng lên line_map
    if lines1 is not None:
        lines1 = lines1[:max_lines] if len(lines1) > max_lines else lines1
        for line in lines1:
            x1, y1, x2, y2 = line[0]
            cv.line(line_map1, (x1, y1), (x2, y2), 1, 2)
    
    if lines2 is not None:
        lines2 = lines2[:max_lines] if len(lines2) > max_lines else lines2
        for line in lines2:
            x1, y1, x2, y2 = line[0]
            cv.line(line_map2, (x1, y1), (x2, y2), 1, 2)
    
    # Làm mịn line_map với kernel nhỏ
    line_map1 = cv.GaussianBlur(line_map1, (3, 3), 0)
    line_map2 = cv.GaussianBlur(line_map2, (3, 3), 0)
    
    # Kết hợp bản đồ đường thẳng vào năng lượng
    line_diff = np.abs(line_map1 - line_map2) * small_overlap.astype(np.float32) / 255.0
    
    # Tính sự khác biệt màu sắc - tối ưu hóa tính toán
    small_overlap_3ch = np.repeat(small_overlap[:, :, np.newaxis] / 255.0, 3, axis=2)
    color_diff = np.sum(np.abs(small_img1.astype(np.float32) - small_img2.astype(np.float32)) * small_overlap_3ch, axis=2) / 3.0
    
    # Tính năng lượng tổng hợp
    combined_energy = np.abs(grad_mag1 - grad_mag2) * 0.3 + color_diff * 0.4 + line_diff * 0.3
    combined_energy = combined_energy * small_overlap.astype(np.float32) / 255.0
    
    # Tính bản đồ khoảng cách
    energy_uint8 = cv.normalize(combined_energy, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    
    # Áp dụng ngưỡng tự động
    _, thresh = cv.threshold(energy_uint8, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    # Tạo bản đồ khoảng cách
    dist_transform = cv.distanceTransform(thresh, cv.DIST_L2, 3)  # Giảm độ chính xác
    
    # Làm mịn bản đồ khoảng cách với kernel nhỏ hơn
    dist_transform = cv.GaussianBlur(dist_transform, (7, 7), 0)  # Giảm kernel từ 11x11 xuống 7x7
    
    # Tạo ngưỡng để tạo mask
    norm_dist = cv.normalize(dist_transform, None, 0, 1.0, cv.NORM_MINMAX)
    _, seam_mask = cv.threshold(norm_dist, 0.5, 1.0, cv.THRESH_BINARY)
    
    # Nếu đã giảm kích thước, resize về kích thước ban đầu
    if max(h, w) > max_dim:
        seam_mask = cv.resize(seam_mask, (w, h), interpolation=cv.INTER_NEAREST)
    
    # Tạo mask cho từng ảnh dựa trên đường ghép
    new_mask1 = mask1.copy()
    new_mask2 = mask2.copy()
    
    # Cập nhật vùng chồng lấp
    overlap_region = (seam_mask * 255).astype(np.uint8) & overlap
    new_mask1[overlap > 0] = overlap_region[overlap > 0]
    new_mask2[overlap > 0] = (255 - overlap_region)[overlap > 0]
    
    # Làm mịn đường ghép
    kernel = np.ones((5,5), np.uint8)
    new_mask1 = cv.morphologyEx(new_mask1, cv.MORPH_CLOSE, kernel)
    new_mask2 = cv.morphologyEx(new_mask2, cv.MORPH_CLOSE, kernel)
    
    return new_mask1, new_mask2

def simple_alpha_blend(img1, img2, mask1, mask2):
    """Alpha blending đơn giản không hiệu chỉnh màu sắc"""
    # Tạo mask cho vùng chồng lấp
    overlap = cv.bitwise_and(mask1, mask2)
    
    if np.sum(overlap) == 0:
        # Không có chồng lấp - ghép trực tiếp
        result = np.zeros_like(img1)
        result = np.where(np.repeat(mask1[:, :, np.newaxis], 3, axis=2) > 0, img1, result)
        result = np.where(np.repeat(mask2[:, :, np.newaxis], 3, axis=2) > 0, img2, result)
        return result
    
    # Làm mịn vùng chồng lấp để tạo chuyển tiếp mượt hơn
    kernel_size = 35  # Tăng kernel size để làm mịn hơn
    overlap_float = overlap.astype(np.float32) / 255.0
    overlap_blur = cv.GaussianBlur(overlap_float, (kernel_size, kernel_size), 0)
    
    # Tính toán gradient alpha từ đường ghép
    # Lấy biên của mask2 để tạo gradient từ biên vào trong
    border_kernel = np.ones((3, 3), np.uint8)
    mask2_border = cv.dilate(mask2, border_kernel) - mask2
    
    # Tạo bản đồ khoảng cách
    dist_transform = cv.distanceTransform((~mask2_border).astype(np.uint8), cv.DIST_L2, 3)
    
    # Chuẩn hóa khoảng cách
    max_dist = np.max(dist_transform) if np.max(dist_transform) > 0 else 1.0
    alpha_map = np.clip(dist_transform / (max_dist * 0.5), 0, 1)
    
    # Trong vùng chồng lấp, alpha biến thiên từ 0.3 đến 0.7 (thay vì 0 đến 1)
    # để hạn chế sự chênh lệch màu quá lớn
    alpha_map = 0.3 + 0.4 * alpha_map
    
    # Làm mịn alpha map
    alpha_map = cv.GaussianBlur(alpha_map, (kernel_size, kernel_size), 0)
    
    # Kết hợp ảnh
    result = np.zeros_like(img1, dtype=np.float32)
    
    # Vùng không chồng lấp
    non_overlap1 = (mask1 > 0) & (overlap == 0)
    non_overlap2 = (mask2 > 0) & (overlap == 0)
    
    result[non_overlap1] = img1[non_overlap1]
    result[non_overlap2] = img2[non_overlap2]
    
    # Vùng chồng lấp - alpha blending với gradient
    overlap_mask = overlap > 0
    alpha_3ch = np.repeat(alpha_map[:, :, np.newaxis], 3, axis=2)
    
    # Nếu alpha_3ch có kích thước khác với img1 hoặc img2, sẽ phải resize
    if alpha_3ch.shape != img1.shape:
        alpha_3ch = cv.resize(alpha_3ch, (img1.shape[1], img1.shape[0]))
    
    # Áp dụng alpha blending
    for i in range(3):  # Xử lý từng kênh màu
        result_channel = result[:,:,i]
        img1_channel = img1[:,:,i]
        img2_channel = img2[:,:,i]
        alpha_channel = alpha_3ch[:,:,i]
        
        # Áp dụng blending
        overlap_indices = np.where(overlap_mask)
        result_channel[overlap_indices] = (
            (1 - alpha_channel[overlap_indices]) * img1_channel[overlap_indices] + 
            alpha_channel[overlap_indices] * img2_channel[overlap_indices]
        )
    
    return np.clip(result, 0, 255).astype(np.uint8)

class QHWStitcher:
    """Lớp thực hiện phương pháp Quasi-Homography Warps (QHW) với lambda thay đổi theo không gian"""
    def __init__(self, alpha=0.6, beta=0.45, gamma=0.65, line_weight=0.3):
        # Các tham số điều chỉnh
        self.alpha = alpha  # Tỷ lệ pha trộn mặc định
        self.beta = beta    # Kiểm soát vùng ảnh hưởng của biên
        self.gamma = gamma  # Kiểm soát độ trơn của hàm trọng số
        self.line_weight = line_weight  # Trọng số cho các cấu trúc đường thẳng
        self.last_model = None
        
        # Điều chỉnh lại các tham số cho cân bằng giữa số lượng và chất lượng
        self.max_keypoints = 200  # Giảm từ 250 xuống 200
        self.downscale_factor = 0.5
    
    def compute_structure_map(self, image):
        """Tạo bản đồ cấu trúc từ ảnh - phiên bản siêu tối ưu tốc độ"""
        h, w = image.shape[:2]
        
        # Giảm kích thước để tính toán nhanh hơn
        max_dim = 200  # Giảm xuống 200
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            small_img = cv.resize(image, None, fx=scale, fy=scale)
        else:
            small_img = image
            
        # Chuyển sang ảnh xám
        gray = cv.cvtColor(small_img, cv.COLOR_BGR2GRAY)
        
        # Tính gradient magnitude
        grad_x = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)  # Giữ kernel 3x3
        grad_y = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)  # Giữ kernel 3x3
        grad_mag = cv.magnitude(grad_x, grad_y)
        
        # Chuẩn hóa
        cv.normalize(grad_mag, grad_mag, 0, 1, cv.NORM_MINMAX)
        
        # Phát hiện đường thẳng - chỉ giới hạn 8 đường
        edges = cv.Canny(gray, 50, 150)
        lines = cv.HoughLinesP(edges, 1, np.pi/180, 
                              threshold=50, 
                              minLineLength=max(gray.shape) // 8,  # Giảm độ dài yêu cầu
                              maxLineGap=15)  # Tăng khoảng cách cho phép
        
        # Tạo bản đồ đường thẳng
        line_map = np.zeros_like(grad_mag)
        if lines is not None:
            # Giới hạn số đường thẳng để xử lý nhanh hơn
            if len(lines) > 8:  # Giảm xuống còn 8 đường
                lines = lines[:8]
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv.line(line_map, (x1, y1), (x2, y2), 1, 2)
                
        # Làm mịn line_map - giữ kernel 3x3
        line_map = cv.GaussianBlur(line_map, (3, 3), 0)
        
        # Kết hợp bản đồ gradient và đường thẳng
        structure_map = np.maximum(grad_mag * 0.7, line_map)
        
        # Làm mịn kết quả cuối cùng - giữ kernel 3x3
        structure_map = cv.GaussianBlur(structure_map, (3, 3), 0)
        
        # Resize về kích thước ban đầu nếu đã thay đổi
        if max(h, w) > max_dim:
            structure_map = cv.resize(structure_map, (w, h))
            
        return structure_map

    def compute_weight_map(self, shape, borders, centers, image=None):
        """Tính bản đồ trọng số λ(x) kết hợp thông tin cấu trúc"""
        h, w = shape[:2]
        
        # Giảm kích thước cho tính toán nhanh hơn
        scale_h, scale_w = int(h * self.downscale_factor), int(w * self.downscale_factor)
        downscale_factor_h, downscale_factor_w = h / scale_h, w / scale_w
        
        # Tạo lưới tọa độ
        y_grid, x_grid = np.mgrid[0:scale_h, 0:scale_w]
        y_grid = y_grid * downscale_factor_h
        x_grid = x_grid * downscale_factor_w
        
        # Tính khoảng cách đến các biên
        dist_left = x_grid
        dist_right = w - 1 - x_grid
        dist_top = y_grid
        dist_bottom = h - 1 - y_grid
        
        # Khoảng cách nhỏ nhất đến biên
        min_dist = np.minimum(np.minimum(dist_left, dist_right), np.minimum(dist_top, dist_bottom))
        
        # Trọng số biên
        max_dist = min(w, h) / 2
        border_weight = np.clip(min_dist / (self.beta * max_dist), 0.0, 1.0)
        
        # Giới hạn số điểm đặc trưng cho tính toán
        if len(centers) > self.max_keypoints:
            centers = centers[np.random.choice(len(centers), self.max_keypoints, replace=False)]
        
        # Khoảng cách đến các điểm đặc trưng
        feature_weight = np.zeros((scale_h, scale_w), dtype=np.float32)
        
        if len(centers) > 0:
            # Vector hóa tính khoảng cách
            centers = np.array(centers)
            # Reshape để broadcasting
            x_expanded = x_grid[:, :, np.newaxis]
            y_expanded = y_grid[:, :, np.newaxis]
            # Tính khoảng cách Euclidean
            distances = np.sqrt(
                (x_expanded - centers[:, 0]) ** 2 + 
                (y_expanded - centers[:, 1]) ** 2
            )
            center_dist = np.min(distances, axis=2)
            # Chuẩn hóa khoảng cách
            feature_weight = np.exp(-(center_dist**2) / (2 * (self.gamma * max(w, h))**2))
        
        # Kết hợp trọng số biên và đặc trưng
        weight_map = self.alpha * border_weight + (1 - self.alpha) * feature_weight
        
        # Thêm thông tin cấu trúc nếu có ảnh đầu vào
        if image is not None:
            # Tính bản đồ cấu trúc
            structure_map = self.compute_structure_map(image)
            
            # Resize structure_map về kích thước của weight_map
            structure_map_resized = cv.resize(structure_map, (scale_w, scale_h))
            
            # Tăng lambda (ưu tiên H) ở các vùng có cấu trúc đường thẳng
            weight_map = np.maximum(weight_map, structure_map_resized * self.line_weight)
        
        # Giới hạn giá trị
        weight_map = np.clip(weight_map, 0.0, 1.0)
        
        # Làm mịn bản đồ trọng số
        sigma = min(scale_w, scale_h) // 30
        if sigma > 0:
            weight_map = cv.GaussianBlur(weight_map, (0, 0), sigmaX=sigma)
        
        # Upscale lại kích thước gốc
        weight_map = cv.resize(weight_map, (w, h))
        
        return weight_map.astype(np.float32)

    def warp_qhw(self, img, H, S, weight_map, output_size):
        """Áp dụng phép biến đổi QHW cho ảnh - phiên bản siêu tối ưu"""
        h, w = img.shape[:2]
        output_h, output_w = output_size
        
        # Nghịch đảo của H và S
        try:
            H_inv = np.linalg.inv(H)
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Xử lý trường hợp ma trận không khả nghịch
            return np.zeros(output_size + (3,), dtype=np.uint8), np.zeros(output_size, dtype=np.uint8)
        
        # Giảm kích thước đầu ra để tính nhanh hơn nếu quá lớn
        scale_factor = 1.0
        if max(output_h, output_w) > 1500:
            scale_factor = 1500 / max(output_h, output_w)
            temp_h, temp_w = int(output_h * scale_factor), int(output_w * scale_factor)
            
            # Điều chỉnh ma trận nghịch đảo
            scale_matrix = np.array([
                [scale_factor, 0, 0],
                [0, scale_factor, 0],
                [0, 0, 1]
            ])
            inv_scale_matrix = np.array([
                [1/scale_factor, 0, 0],
                [0, 1/scale_factor, 0],
                [0, 0, 1]
            ])
            H_inv = inv_scale_matrix @ H_inv @ scale_matrix
            S_inv = inv_scale_matrix @ S_inv @ scale_matrix
            
            output_h, output_w = temp_h, temp_w
        
        # Tạo kết quả cuối cùng
        result = np.zeros((output_h, output_w, 3), dtype=np.uint8)
        mask = np.zeros((output_h, output_w), dtype=np.uint8)
        
        # Kích thước tile cho đa luồng - tăng kích thước để giảm overhead
        tile_size = 256  # Tăng từ 128 lên 256
        
        import concurrent.futures
        import threading
        
        # Lock để đồng bộ hóa truy cập vào kết quả từ các thread
        result_lock = threading.Lock()
        
        # Hàm tối ưu để xử lý một khối
        def process_tile(y_start, y_end, x_start, x_end):
            # Tạo lưới tọa độ cho khối
            y_grid, x_grid = np.mgrid[y_start:y_end, x_start:x_end]
            
            # Biến đổi sang dạng vector để tính nhanh
            points = np.ones((3, (y_end - y_start) * (x_end - x_start)), dtype=np.float32)
            points[0, :] = x_grid.flatten()
            points[1, :] = y_grid.flatten()
            
            # Áp dụng biến đổi ngược
            points_h = np.dot(H_inv, points)
            points_s = np.dot(S_inv, points)
            
            # Chuẩn hóa tọa độ
            mask_h = points_h[2, :] != 0
            mask_s = points_s[2, :] != 0
            
            # Tránh chia cho 0
            if np.any(mask_h):
                points_h[:2, mask_h] /= points_h[2, mask_h]
            if np.any(mask_s):
                points_s[:2, mask_s] /= points_s[2, mask_s]
            
            # Reshape lại tọa độ
            x_h = points_h[0, :].reshape(y_end - y_start, x_end - x_start)
            y_h = points_h[1, :].reshape(y_end - y_start, x_end - x_start)
            x_s = points_s[0, :].reshape(y_end - y_start, x_end - x_start)
            y_s = points_s[1, :].reshape(y_end - y_start, x_end - x_start)
            
            # Tạo kết quả cục bộ
            local_result = np.zeros((y_end - y_start, x_end - x_start, 3), dtype=np.uint8)
            local_mask = np.zeros((y_end - y_start, x_end - x_start), dtype=np.uint8)
            
            # Chỉ xét các điểm trong khối
            valid_mask = (
                (x_h >= 0) & (x_h < w-1) & (y_h >= 0) & (y_h < h-1) &
                (x_s >= 0) & (x_s < w-1) & (y_s >= 0) & (y_s < h-1)
            )
            
            # Tìm các điểm hợp lệ
            y_valid, x_valid = np.where(valid_mask)
            
            # Xử lý từng lô điểm để tối ưu bộ nhớ
            BATCH_SIZE = 25000  # Tăng từ 10000 lên 25000
            num_points = len(y_valid)
            
            for batch_start in range(0, num_points, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, num_points)
                batch_y = y_valid[batch_start:batch_end]
                batch_x = x_valid[batch_start:batch_end]
                
                # Lấy tọa độ trong ảnh gốc cho lô điểm
                batch_x_h = x_h[batch_y, batch_x]
                batch_y_h = y_h[batch_y, batch_x]
                batch_x_s = x_s[batch_y, batch_x]
                batch_y_s = y_s[batch_y, batch_x]
                
                # Lấy lambda (trọng số) từ weight map
                batch_lambda = np.zeros(batch_end - batch_start, dtype=np.float32)
                
                # Vector hóa hơn nữa việc lấy lambda
                valid_y_h = np.clip(batch_y_h.astype(np.int32), 0, h-1)
                valid_x_h = np.clip(batch_x_h.astype(np.int32), 0, w-1)
                in_bounds = (0 <= batch_y_h) & (batch_y_h < h) & (0 <= batch_x_h) & (batch_x_h < w)
                batch_lambda[in_bounds] = weight_map[valid_y_h[in_bounds], valid_x_h[in_bounds]]
                batch_lambda[~in_bounds] = self.alpha
                
                # Kết hợp tọa độ
                batch_x_src = batch_lambda * batch_x_h + (1 - batch_lambda) * batch_x_s
                batch_y_src = batch_lambda * batch_y_h + (1 - batch_lambda) * batch_y_s
                
                # Tính nội suy màu cho các điểm trong lô
                batch_x_src_int = np.clip(batch_x_src.astype(np.int32), 0, w-2)
                batch_y_src_int = np.clip(batch_y_src.astype(np.int32), 0, h-2)
                batch_x_src_frac = batch_x_src - batch_x_src_int
                batch_y_src_frac = batch_y_src - batch_y_src_int
                
                # Lấy 4 điểm lân cận cho từng pixel (vector hóa)
                c00 = img[batch_y_src_int, batch_x_src_int]
                c01 = img[batch_y_src_int, batch_x_src_int + 1]
                c10 = img[batch_y_src_int + 1, batch_x_src_int]
                c11 = img[batch_y_src_int + 1, batch_x_src_int + 1]
                
                # Tính trọng số nội suy
                w00 = (1 - batch_x_src_frac) * (1 - batch_y_src_frac)
                w01 = batch_x_src_frac * (1 - batch_y_src_frac)
                w10 = (1 - batch_x_src_frac) * batch_y_src_frac
                w11 = batch_x_src_frac * batch_y_src_frac
                
                # Nội suy màu (vector hóa)
                w00 = w00[:, np.newaxis]
                w01 = w01[:, np.newaxis]
                w10 = w10[:, np.newaxis]
                w11 = w11[:, np.newaxis]
                
                interpolated_colors = w00 * c00 + w01 * c01 + w10 * c10 + w11 * c11
                
                # Gán màu vào kết quả cục bộ
                local_result[batch_y, batch_x] = interpolated_colors.astype(np.uint8)
                local_mask[batch_y, batch_x] = 255
            
            # Cập nhật kết quả chung
            with result_lock:
                result[y_start:y_end, x_start:x_end] = local_result
                mask[y_start:y_end, x_start:x_end] = local_mask
        
        # Tính số tile và chia công việc
        num_tiles_h = (output_h + tile_size - 1) // tile_size
        num_tiles_w = (output_w + tile_size - 1) // tile_size
        
        # Tối ưu số thread
        import os
        num_cores = min(os.cpu_count() or 4, 8)  # Giới hạn số thread tối đa 8
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
            futures = []
            
            for ty in range(num_tiles_h):
                for tx in range(num_tiles_w):
                    y_start = ty * tile_size
                    y_end = min(output_h, (ty + 1) * tile_size)
                    x_start = tx * tile_size
                    x_end = min(output_w, (tx + 1) * tile_size)
                    
                    futures.append(executor.submit(process_tile, y_start, y_end, x_start, x_end))
            
            concurrent.futures.wait(futures)
        
        # Upscale lại kết quả nếu đã giảm kích thước
        if scale_factor < 1.0:
            orig_h, orig_w = output_size
            result = cv.resize(result, (orig_w, orig_h), interpolation=cv.INTER_LINEAR)
            mask = cv.resize(mask, (orig_w, orig_h), interpolation=cv.INTER_LINEAR)
            # Chuyển lại về binary mask
            mask = np.where(mask > 127, 255, 0).astype(np.uint8)
        
        return result, mask

    def compute_similarity_transform(self, src_pts, dst_pts):
        """Tính ma trận similarity transform giữa hai tập hợp điểm - đã vector hóa"""
        # Giới hạn số điểm nếu quá nhiều
        if len(src_pts) > self.max_keypoints:
            # Lấy mẫu ngẫu nhiên từ các điểm
            indices = np.random.choice(len(src_pts), self.max_keypoints, replace=False)
            src_pts = src_pts[indices]
            dst_pts = dst_pts[indices]
            
        # Tính tâm của các điểm
        src_mean = np.mean(src_pts, axis=0)
        dst_mean = np.mean(dst_pts, axis=0)
        
        # Dịch chuyển các điểm về gốc tọa độ
        src_centered = src_pts - src_mean
        dst_centered = dst_pts - dst_mean
        
        # Tính ma trận hiệp phương sai
        cov_matrix = np.dot(dst_centered.T, src_centered)
        
        # Phân rã SVD
        U, _, Vt = np.linalg.svd(cov_matrix)
        
        # Tính ma trận quay
        rotation = np.dot(U, Vt)
        
        # Đảm bảo không phản chiếu
        if np.linalg.det(rotation) < 0:
            Vt[-1, :] *= -1
            rotation = np.dot(U, Vt)
        
        # Tính tỷ lệ
        scale = np.sum(dst_centered**2) / np.sum(src_centered**2)
        scale = np.sqrt(scale)
        
        # Xây dựng ma trận similarity transform
        S = np.eye(3, dtype=np.float32)
        S[:2, :2] = scale * rotation
        S[0, 2] = dst_mean[0] - scale * np.dot(rotation[0], src_mean)
        S[1, 2] = dst_mean[1] - scale * np.dot(rotation[1], src_mean)
        
        return S
    
    def find_key_borders(self, src_pts, img_shape):
        """Xác định vùng biên quan trọng dựa trên điểm đặc trưng - đã vector hóa"""
        h, w = img_shape[:2]
        
        # Giới hạn số điểm nếu quá nhiều
        if len(src_pts) > 1000:
            indices = np.random.choice(len(src_pts), 1000, replace=False)
            src_pts = src_pts[indices]
        
        # Tính tọa độ bao quanh các điểm đặc trưng
        min_x = max(0, np.min(src_pts[:, 0]))
        min_y = max(0, np.min(src_pts[:, 1]))
        max_x = min(w-1, np.max(src_pts[:, 0]))
        max_y = min(h-1, np.max(src_pts[:, 1]))
        
        # Mở rộng vùng quan trọng
        padding_x = (max_x - min_x) * 0.1
        padding_y = (max_y - min_y) * 0.1
        
        min_x = max(0, min_x - padding_x)
        min_y = max(0, min_y - padding_y)
        max_x = min(w-1, max_x + padding_x)
        max_y = min(h-1, max_y + padding_y)
        
        return (min_x, min_y, max_x, max_y)
    
    def stitch(self, good_matches, kp1, kp2, img1, img2):
        """Ghép ảnh sử dụng phương pháp QHW với cân bằng chất lượng và tốc độ"""
        threshold = dynamic_good_match_threshold(kp1, kp2)
        multiplier = failure_tracker.get_threshold_multiplier()
        threshold = int(threshold * multiplier)
        
        # Điều chỉnh lại ngưỡng điểm tối thiểu
        if len(good_matches) < max(6, threshold // 3):  # Thay đổi từ threshold/2 thành threshold/3
            failure_tracker.update(False)
            print(f"Not enough matches ({len(good_matches)}) vs adjusted threshold ({max(6, threshold // 3)})")
            return None
        else:
            failure_tracker.update(True)
        
        # Đảm bảo luôn có ít nhất 4 điểm để tính homography
        if len(good_matches) < 4:
            failure_tracker.update(False)
            print(f"Not enough matches ({len(good_matches)}) to compute homography")
            return None
        
        # Trích xuất điểm tương ứng
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        
        # Giảm kích thước ảnh để tính toán nhanh hơn nếu quá lớn
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Chỉ giảm kích thước nếu ảnh quá lớn
        max_dimension = 700
        if max(h1, w1, h2, w2) > max_dimension:
            scale_factor = max_dimension / max(h1, w1, h2, w2)
            # Giảm kích thước cho phép tính toán
            small_img1 = cv.resize(img1, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_AREA)
            small_h1, small_w1 = small_img1.shape[:2]
            
            # Điều chỉnh tọa độ điểm theo tỷ lệ
            small_src_pts = src_pts * scale_factor
            small_dst_pts = dst_pts * scale_factor
        else:
            # Sử dụng kích thước gốc
            small_img1 = img1
            small_h1, small_w1 = h1, w1
            small_src_pts = src_pts
            small_dst_pts = dst_pts
        
        # Lọc outliers bằng RANSAC với độ chính xác tốt hơn
        H_global, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 4.5)
        
        if mask is None:
            print("Failed to compute homography")
            return None
            
        # Chỉ giữ lại inliers
        inlier_mask = mask.ravel() == 1
        inlier_count = np.sum(inlier_mask)
        print(f"Inliers: {inlier_count}/{len(mask)}")
        
        # Kiểm tra tỷ lệ inlier quá thấp
        if inlier_count / len(mask) < 0.4:  # Thêm kiểm tra này
            print(f"Inlier ratio too low: {inlier_count}/{len(mask)} = {inlier_count/len(mask):.2f}")
            return None
            
        src_pts_inlier = src_pts[inlier_mask]
        dst_pts_inlier = dst_pts[inlier_mask]
        
        # Yêu cầu ít nhất 6 điểm inlier (thay vì 4)
        if len(src_pts_inlier) < 6:
            print(f"Not enough inliers: {len(src_pts_inlier)}")
            return None
            
        # Tính similarity transformation
        S = self.compute_similarity_transform(src_pts_inlier, dst_pts_inlier)
        
        # Kiểm tra matrix homography và similarity có hợp lệ không
        # Ma trận hợp lệ không được có giá trị NaN hoặc Inf
        if (np.any(np.isnan(H_global)) or np.any(np.isinf(H_global)) or 
            np.any(np.isnan(S)) or np.any(np.isinf(S))):
            print("Invalid transformation matrix detected")
            return None
            
        # Tìm biên quan trọng
        borders = self.find_key_borders(src_pts_inlier, small_img1.shape)
        
        # Sửa lỗi reshape
        corners = np.array([[0, 0, 1], 
                           [w1, 0, 1], 
                           [w1, h1, 1], 
                           [0, h1, 1]], dtype=np.float32)
        
        # Biến đổi góc bằng homography và similarity
        corners_H = np.zeros((4, 2), dtype=np.float32)
        corners_S = np.zeros((4, 2), dtype=np.float32)
        
        for i in range(4):
            # Biến đổi bằng homography
            p_h = np.dot(H_global, corners[i])
            corners_H[i] = [p_h[0]/p_h[2], p_h[1]/p_h[2]]
            
            # Biến đổi bằng similarity
            p_s = np.dot(S, corners[i])
            corners_S[i] = [p_s[0]/p_s[2], p_s[1]/p_s[2]]
        
        # Kết hợp các góc để tính kích thước tối đa
        all_corners = np.vstack([corners_H, corners_S, [[0, 0], [w2, 0], [w2, h2], [0, h2]]])
        
        # Loại bỏ các outlier trong corners - sử dụng phương pháp IQR
        x_coords = all_corners[:, 0]
        y_coords = all_corners[:, 1]
        
        # Tính IQR cho x và y
        x_q1, x_q3 = np.percentile(x_coords, [25, 75])
        y_q1, y_q3 = np.percentile(y_coords, [25, 75])
        
        x_iqr = x_q3 - x_q1
        y_iqr = y_q3 - y_q1
        
        # Lọc các điểm nằm trong phạm vi 1.5 IQR
        inlier_corners = all_corners[
            (x_coords >= x_q1 - 1.5 * x_iqr) & 
            (x_coords <= x_q3 + 1.5 * x_iqr) & 
            (y_coords >= y_q1 - 1.5 * y_iqr) & 
            (y_coords <= y_q3 + 1.5 * y_iqr)
        ]
        
        # Nếu không có inlier nào (hiếm khi xảy ra), sử dụng tất cả corners
        if len(inlier_corners) == 0:
            inlier_corners = all_corners
            
        # Tính bounding box từ inlier corners
        x_min, y_min = np.min(inlier_corners, axis=0)
        x_max, y_max = np.max(inlier_corners, axis=0)
        
        # Tính ma trận dịch chuyển
        translation = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Điều chỉnh các ma trận biến đổi
        H_global_adjusted = translation @ H_global
        S_adjusted = translation @ S
        
        # Kích thước ảnh đích
        output_width = int(x_max - x_min)
        output_height = int(y_max - y_min)
        
        # Giới hạn kích thước tối đa
        MAX_CANVAS_WIDTH = 3000
        MAX_CANVAS_HEIGHT = 3000
        if output_width > MAX_CANVAS_WIDTH or output_height > MAX_CANVAS_HEIGHT:
            # Giảm kích thước nếu quá lớn
            scale = min(MAX_CANVAS_WIDTH / output_width, MAX_CANVAS_HEIGHT / output_height)
            output_width = int(output_width * scale)
            output_height = int(output_height * scale)
            
            # Điều chỉnh ma trận biến đổi
            scale_matrix = np.array([
                [scale, 0, 0],
                [0, scale, 0],
                [0, 0, 1]
            ])
            H_global_adjusted = scale_matrix @ H_global_adjusted
            S_adjusted = scale_matrix @ S_adjusted
            
        output_size = (output_height, output_width)
        
        # Tính bản đồ trọng số với thông tin cấu trúc
        # Giới hạn sử dụng cấu trúc để tránh biến dạng quá mức
        use_structure = max(h1, w1) <= 700  # Giảm xuống
        
        if use_structure:
            weight_map = self.compute_weight_map(small_img1.shape, borders, small_src_pts, small_img1)
        else:
            weight_map = self.compute_weight_map(small_img1.shape, borders, small_src_pts)
        
        print("Đang áp dụng QHW warping...")
        t_start = cv.getTickCount()
        
        # Biến đổi ảnh thứ nhất bằng QHW
        warped_img1, mask1 = self.warp_qhw(img1, H_global_adjusted, S_adjusted, weight_map, output_size)
        
        t_end = cv.getTickCount()
        time_taken = (t_end - t_start) / cv.getTickFrequency()
        print(f"Warping hoàn thành trong {time_taken:.2f} giây")
        
        # Đặt ảnh thứ hai vào vị trí
        warped_img2 = np.zeros(output_size + (3,), dtype=np.uint8)
        mask2 = np.zeros((output_height, output_width), dtype=np.uint8)
        
        # Copy ảnh thứ hai vào canvas đích với phép dịch chuyển
        x_offset, y_offset = int(-x_min), int(-y_min)
        
        # Kiểm tra phạm vi hợp lệ
        if (x_offset >= 0 and y_offset >= 0 and 
            x_offset + w2 <= output_width and y_offset + h2 <= output_height):
            warped_img2[y_offset:y_offset+h2, x_offset:x_offset+w2] = img2
            mask2[y_offset:y_offset+h2, x_offset:x_offset+w2] = 255
        else:
            print(f"Warning: Second image outside canvas (offset: {x_offset}, {y_offset}, size: {output_width}x{output_height}, img: {w2}x{h2})")
            # Tính toán phần có thể copy
            x_start = max(0, x_offset)
            y_start = max(0, y_offset)
            x_end = min(output_width, x_offset + w2)
            y_end = min(output_height, y_offset + h2)
            
            # Tính vị trí tương ứng trong ảnh nguồn
            x_src_start = max(0, -x_offset)
            y_src_start = max(0, -y_offset)
            x_src_end = x_src_start + (x_end - x_start)
            y_src_end = y_src_start + (y_end - y_start)
            
            # Copy phần hợp lệ
            if x_end > x_start and y_end > y_start and x_src_end > x_src_start and y_src_end > y_src_start:
                warped_img2[y_start:y_end, x_start:x_end] = img2[y_src_start:y_src_end, x_src_start:x_src_end]
                mask2[y_start:y_end, x_start:x_end] = 255
        
        print("Tìm đường ghép tối ưu và trộn ảnh...")
        
        try:
            # Tìm đường ghép tối ưu
            seam_mask1, seam_mask2 = find_optimal_seam(warped_img1, warped_img2, mask1, mask2)
            
            # Trộn ảnh với alpha blending đơn giản không hiệu chỉnh màu
            result = simple_alpha_blend(warped_img1, warped_img2, seam_mask1, seam_mask2)
            
            # Cắt bỏ viền đen thừa
            result = fast_crop_black_borders(result)
            
            print("Hoàn thành ghép ảnh!")
            
            # Lưu mô hình hiện tại
            self.last_model = {
                'H': H_global_adjusted,
                'S': S_adjusted,
                'weight_map': weight_map
            }
            
            return result
        except Exception as e:
            print(f"Error in stitching process: {e}")
            return None

def color_correct_images(img1, img2, mask1, mask2):
    """Hiệu chỉnh màu sắc triệt để giữa hai ảnh"""
    # Tạo mask cho vùng chồng lấp
    overlap = cv.bitwise_and(mask1, mask2)
    
    if np.sum(overlap) == 0:
        return img1, img2
    
    # Tách các kênh màu
    img1_lab = cv.cvtColor(img1, cv.COLOR_BGR2LAB)
    img2_lab = cv.cvtColor(img2, cv.COLOR_BGR2LAB)
    
    # Điều chỉnh cho từng kênh trong không gian LAB
    for i in range(3):
        # Tính giá trị trung bình và độ lệch chuẩn trong vùng chồng lấp
        mean1 = cv.mean(img1_lab[:,:,i], mask=overlap)[0]
        mean2 = cv.mean(img2_lab[:,:,i], mask=overlap)[0]
        
        # Tính độ lệch chuẩn
        std1 = np.sqrt(cv.mean((img1_lab[:,:,i].astype(np.float32) - mean1)**2, mask=overlap)[0])
        std2 = np.sqrt(cv.mean((img2_lab[:,:,i].astype(np.float32) - mean2)**2, mask=overlap)[0])
        
        # Điều chỉnh độ sáng và độ tương phản
        if std2 != 0:
            gain = std1 / std2
        else:
            gain = 1.0
        
        # Giới hạn gain để tránh thay đổi quá mức nhưng cho phép điều chỉnh mạnh hơn
        gain = np.clip(gain, 0.5, 2.0)
        
        # Tính offset (độ lệch)
        offset = mean1 - gain * mean2
        
        # Áp dụng điều chỉnh cho toàn bộ ảnh thứ hai
        img2_lab[:,:,i] = np.clip(img2_lab[:,:,i].astype(np.float32) * gain + offset, 0, 255).astype(np.uint8)
    
    # Chuyển lại sang không gian BGR
    adjusted_img2 = cv.cvtColor(img2_lab, cv.COLOR_LAB2BGR)
    
    # Tạo mặt nạ chuyển tiếp mượt để phối trộn điều chỉnh màu
    y, x = np.indices(mask2.shape)
    h, w = mask2.shape
    
    # Tính khoảng cách từ mỗi điểm đến biên gần nhất
    dist_from_border = np.minimum.reduce([
        x,  # khoảng cách đến biên trái
        y,  # khoảng cách đến biên trên
        w - x - 1,  # khoảng cách đến biên phải
        h - y - 1,  # khoảng cách đến biên dưới
    ])
    
    # Tạo hệ số trộn dựa trên khoảng cách (càng xa biên, điều chỉnh càng nhiều)
    blend_factor = np.minimum(1.0, dist_from_border / 50.0)
    blend_factor = np.clip(blend_factor, 0, 1)
    
    # Áp dụng điều chỉnh màu thủ công thay vì sử dụng cv.addWeighted
    blended_img2 = np.zeros_like(img2)
    
    # Mở rộng blend_factor để phù hợp với 3 kênh màu
    blend_factor_3ch = np.repeat(blend_factor[:, :, np.newaxis], 3, axis=2)
    
    # Công thức: (1 - alpha) * img2 + alpha * adjusted_img2
    blended_img2 = ((1 - blend_factor_3ch) * img2 + 
                    blend_factor_3ch * adjusted_img2).astype(np.uint8)
    
    return img1, blended_img2

def color_correct_lab(img1, img2, mask1, mask2):
    """Hiệu chỉnh màu sắc cải tiến sử dụng không gian màu LAB"""
    # Tạo mask cho vùng chồng lấp
    overlap = cv.bitwise_and(mask1, mask2)
    
    if np.sum(overlap) == 0:
        return img1, img2
    
    # Chuyển sang không gian LAB
    img1_lab = cv.cvtColor(img1, cv.COLOR_BGR2LAB)
    img2_lab = cv.cvtColor(img2, cv.COLOR_BGR2LAB)
    
    # Tạo một bản sao của img2_lab để hiệu chỉnh
    adjusted_img2_lab = img2_lab.copy().astype(np.float32)
    
    # Điều chỉnh cho từng kênh trong không gian LAB
    for i in range(3):
        # Tính giá trị trung bình và độ lệch chuẩn trong vùng chồng lấp
        mean1 = cv.mean(img1_lab[:,:,i], mask=overlap)[0]
        mean2 = cv.mean(img2_lab[:,:,i], mask=overlap)[0]
        
        # Tính độ lệch chuẩn
        std1 = np.sqrt(cv.mean((img1_lab[:,:,i].astype(np.float32) - mean1)**2, mask=overlap)[0])
        std2 = np.sqrt(cv.mean((img2_lab[:,:,i].astype(np.float32) - mean2)**2, mask=overlap)[0])
        
        # Điều chỉnh độ sáng và độ tương phản
        gain = std1 / max(std2, 1e-5)  # Tránh chia cho 0
        
        # Giới hạn gain để tránh thay đổi quá mức
        gain = np.clip(gain, 0.7, 1.5)
        
        # Tính offset
        offset = mean1 - gain * mean2
        
        # Áp dụng điều chỉnh
        adjusted_img2_lab[:,:,i] = adjusted_img2_lab[:,:,i] * gain + offset
    
    # Clip giá trị để tránh overflow
    adjusted_img2_lab = np.clip(adjusted_img2_lab, 0, 255).astype(np.uint8)
    
    # Chuyển trở lại không gian BGR
    adjusted_img2 = cv.cvtColor(adjusted_img2_lab, cv.COLOR_LAB2BGR)
    
    # Tạo mặt nạ chuyển tiếp mượt cho blending
    y, x = np.indices(mask2.shape)
    h, w = mask2.shape
    
    # Tính khoảng cách đến biên
    dist_from_border = np.minimum.reduce([
        x, y, w - x - 1, h - y - 1
    ])
    
    # Tính alpha blend dựa trên khoảng cách
    max_distance = min(w, h) / 4  # Tăng phạm vi ảnh hưởng
    blend_factor = np.minimum(1.0, dist_from_border / max_distance)
    blend_factor = np.clip(blend_factor, 0, 1)
    
    # Mở rộng blend_factor cho 3 kênh
    blend_factor_3ch = np.repeat(blend_factor[:, :, np.newaxis], 3, axis=2)
    
    # Blend ảnh gốc và ảnh đã điều chỉnh
    blended_img2 = ((1 - blend_factor_3ch) * img2 + 
                   blend_factor_3ch * adjusted_img2).astype(np.uint8)
    
    return img1, blended_img2

def improved_blend(img1, img2, mask1, mask2):
    """Trộn hai ảnh với alpha blending cải tiến"""
    # Tạo mask chồng lấp
    overlap = cv.bitwise_and(mask1, mask2)
    
    if np.sum(overlap) == 0:
        # Không có chồng lấp - ghép trực tiếp
        result = np.zeros_like(img1)
        result = np.where(np.repeat(mask1[:, :, np.newaxis], 3, axis=2) > 0, img1, result)
        result = np.where(np.repeat(mask2[:, :, np.newaxis], 3, axis=2) > 0, img2, result)
        return result
    
    # Làm mịn vùng chồng lấp để tạo chuyển tiếp mượt hơn
    kernel_size = 15  # Tăng lên từ 9 để làm mịn hơn
    overlap_float = overlap.astype(np.float32) / 255.0
    overlap_blur = cv.GaussianBlur(overlap_float, (kernel_size, kernel_size), 0)
    
    # Tạo mask 3 kênh
    mask1_3ch = np.repeat(mask1[:, :, np.newaxis] / 255.0, 3, axis=2)
    mask2_3ch = np.repeat(mask2[:, :, np.newaxis] / 255.0, 3, axis=2)
    overlap_3ch = np.repeat(overlap_blur[:, :, np.newaxis], 3, axis=2)
    
    # Tạo gradient alpha từ đường ghép
    # Dựa trên khoảng cách đến biên ảnh
    y, x = np.indices(mask1.shape)
    h, w = mask1.shape
    
    # Tính khoảng cách từ mỗi pixel đến biên gần nhất của img2
    dist_map = np.zeros_like(mask1, dtype=np.float32)
    
    # Lấy biên của mask2
    mask2_border = cv.dilate(mask2, np.ones((3,3), np.uint8)) - mask2
    
    # Tính khoảng cách
    dist_transform = cv.distanceTransform((~mask2_border).astype(np.uint8), cv.DIST_L2, 3)
    
    # Chuẩn hóa khoảng cách
    max_dist = np.max(dist_transform) if np.max(dist_transform) > 0 else 1.0
    normalized_dist = dist_transform / max_dist
    
    # Tạo alpha dựa trên khoảng cách và vùng chồng lấp
    alpha = np.zeros_like(overlap_float)
    alpha[overlap > 0] = normalized_dist[overlap > 0]
    alpha = cv.GaussianBlur(alpha, (kernel_size, kernel_size), 0)
    
    # Kết hợp ảnh
    result = np.zeros_like(img1, dtype=np.float32)
    
    # Vùng không chồng lấp
    non_overlap1 = (mask1 > 0) & (overlap == 0)
    non_overlap2 = (mask2 > 0) & (overlap == 0)
    
    result[non_overlap1] = img1[non_overlap1]
    result[non_overlap2] = img2[non_overlap2]
    
    # Vùng chồng lấp - sử dụng alpha
    overlap_mask = overlap > 0
    alpha_3ch = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
    result[overlap_mask] = (img1[overlap_mask] * alpha_3ch[overlap_mask] + 
                          img2[overlap_mask] * (1 - alpha_3ch[overlap_mask]))
    
    return result.astype(np.uint8)

# Khởi tạo QHW Stitcher với lambda thay đổi theo không gian
qhw_stitcher = QHWStitcher(alpha=0.6, beta=0.45, gamma=0.65, line_weight=0.3)

def image_stitching(image, ShowMatcher=False):
    """Ghép ảnh tối ưu tốc độ cho ứng dụng real-time với chất lượng cân bằng"""
    global region_memory, prev_images
    
    # Kiểm tra chuyển động - sử dụng optical flow được tối ưu
    needs_update = False
    for idx, img in enumerate(image):
        img_key = f"camera_{idx}"
        if img_key in prev_images:
            # Phát hiện chuyển động với optical flow tối ưu
            if opticalflow_motion_detection(img, prev_images[img_key], motion_threshold=0.5, min_moving_points_ratio=0.05):
                needs_update = True
        else:
            needs_update = True
        prev_images[img_key] = img.copy()
    
    # Nếu không có chuyển động đáng kể và đã có kết quả trước đó
    if not needs_update and region_memory.panorama_history is not None:
        return region_memory.panorama_history
    
    # Giảm kích thước ảnh để tăng tốc xử lý cho ứng dụng real-time
    max_dimension_processing = 500
    
    original_sizes = []
    downscaled_images = []
    
    for img in image:
        h, w = img.shape[:2]
        original_sizes.append((h, w))
        
        if max(h, w) > max_dimension_processing:
            scale = max_dimension_processing / max(h, w)
            downscaled = cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
            downscaled_images.append(downscaled)
        else:
            downscaled_images.append(img)
    
    # Sử dụng SIFT với số feature cân bằng giữa tốc độ và chất lượng
    kp = []
    des = []
    for img in downscaled_images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Giảm số lượng đặc trưng xuống, tăng độ chính xác
        sift = cv.SIFT_create(nfeatures=700, contrastThreshold=0.03, edgeThreshold=15)
        k, d = sift.detectAndCompute(gray, None)
        kp.append(k)
        des.append(d)
    
    # Sử dụng matcher với ngưỡng cân bằng
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    if des[0] is not None and des[1] is not None and len(des[0]) >= 2 and len(des[1]) >= 2:
        matches = matcher.knnMatch(des[0], des[1], 2)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                # Giảm ngưỡng từ 0.8 xuống 0.75 để chỉ lấy điểm chất lượng tốt hơn
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
    else:
        good_matches = []
        
    # Yêu cầu ít nhất 8 điểm
    if len(good_matches) < 8:
        print(f"Không tìm thấy đủ điểm trùng khớp: {len(good_matches)}")
        if region_memory.panorama_history is not None:
            return region_memory.panorama_history
        return None
    
    print(f"Tìm thấy {len(good_matches)} điểm trùng khớp")
    
    # Kiểm tra phân bố không gian của các điểm
    if len(good_matches) > 10:
        # Lấy tọa độ các điểm trùng khớp
        pts1 = np.float32([kp[0][m.queryIdx].pt for m in good_matches])
        
        # Tính toán hình chữ nhật bao quanh các điểm
        x_min, y_min = np.min(pts1, axis=0)
        x_max, y_max = np.max(pts1, axis=0)
        width, height = x_max - x_min, y_max - y_min
        img_width, img_height = downscaled_images[0].shape[1], downscaled_images[0].shape[0]
        
        # Kiểm tra xem các điểm có phân bố trên ít nhất 40% diện tích ảnh không
        coverage_ratio = (width * height) / (img_width * img_height)
        if coverage_ratio < 0.1:  # Phân bố quá tập trung
            print(f"Điểm trùng khớp phân bố quá tập trung: {coverage_ratio:.2f}")
            if region_memory.panorama_history is not None:
                return region_memory.panorama_history
            return None
    
    if ShowMatcher:
        show_matcher(downscaled_images[0], kp[0], downscaled_images[1], kp[1], good_matches)

    # Sử dụng QHW Stitcher để ghép ảnh
    result = qhw_stitcher.stitch(good_matches, kp[0], kp[1], downscaled_images[0], downscaled_images[1])

    if result is None:
        print("Ghép ảnh thất bại")
        if region_memory.panorama_history is not None:
            return region_memory.panorama_history
        return None
    
    # Cắt ảnh với hàm tối ưu
    result = fast_crop_black_borders(result)
    
    # Cập nhật vào bộ nhớ
    region_memory.panorama_history = result.copy()
    
    return result

if __name__ == "__main__":
    import time
    img1 = cv.imread('D:/CPV_project/stitch-env/6_1.jpg')
    img2 = cv.imread('D:/CPV_project/stitch-env/6_2.jpg')
    
    if img1 is None or img2 is None:
        print("Cannot read images")
        exit()
    
    # Giảm kích thước ảnh cho demo real-time nhưng vẫn giữ chất lượng tốt
    max_dimension = 900  # Tăng từ 800 lên 900
    if img1.shape[1] > max_dimension:
        scale = max_dimension / img1.shape[1]
        img1 = cv.resize(img1, None, fx=scale, fy=scale)
        img2 = cv.resize(img2, None, fx=scale, fy=scale)
    
    # Đo thời gian
    start_time = time.time()
    result = image_stitching([img1, img2])
    end_time = time.time()
    
    print(f"Tổng thời gian xử lý: {end_time - start_time:.2f} giây")
    
    if result is not None:
        cv.imshow('Stitched Image', result)
        cv.waitKey(0)
        cv.destroyAllWindows()
