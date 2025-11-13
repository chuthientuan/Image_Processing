import streamlit as st
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter
import cv2

# --- CẤU HÌNH TRANG (PHẢI LÀ LỆNH STREAMLIT ĐẦU TIÊN) ---
st.set_page_config(layout="wide")


# --- TẤT CẢ CÁC HÀM XỬ LÝ (Thủ công VÀ OPENCV) ---

# --- Các hàm Thủ công (từ code của bạn) ---
def remove_shadow_manual(gray_img, kernel_size=15, sigma=5, eps=1e-3):
    k = kernel_size // 2
    x, y = np.mgrid[-k:k + 1, -k:k + 1]
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    H, W = gray_img.shape
    padded = np.pad(gray_img, ((k, k), (k, k)), mode='reflect')
    blurred = np.zeros_like(gray_img, dtype=np.float32)
    for i in range(kernel_size):
        for j in range(kernel_size):
            blurred += kernel[i, j] * padded[i:i + H, j:j + W]
    shadow_free = gray_img.astype(np.float32) / (blurred.astype(np.float32) / 255.0 + eps)
    return np.clip(shadow_free, 0, 255).astype(np.uint8)


def hist_equalize_manual(gray_img):
    hist = np.zeros(256, dtype=int)
    for v in gray_img.flatten(): hist[v] += 1
    cdf = np.cumsum(hist)
    # Xử lý trường hợp ảnh trống hoặc ảnh có 1 màu duy nhất
    cdf_min_val = cdf[cdf > 0]
    if len(cdf_min_val) == 0:
        return gray_img  # Trả về ảnh gốc nếu không có gì để cân bằng
    cdf_min = cdf_min_val[0]

    total_pixels = gray_img.size
    # Tránh chia cho 0
    denominator = total_pixels - cdf_min
    if denominator == 0:
        return gray_img

    lut = ((cdf - cdf_min) * 255 / denominator).clip(0, 255).astype(np.uint8)
    return lut[gray_img]


def global_hist_threshold_manual(image):
    hist = np.zeros(256, dtype=np.float32)
    for val in image.ravel(): hist[val] += 1
    hist_sum = hist.sum()
    if hist_sum == 0:
        return (image < 128).astype(np.uint8) * 255  # Trả về mặc định nếu ảnh trống
    hist /= hist_sum

    bins = np.arange(256)
    total_mean = np.sum(bins * hist)
    best_thresh, max_between = 0, -1
    w0_acc, sum0_acc = 0.0, 0.0
    for t in range(256):
        w0_acc += hist[t];
        sum0_acc += t * hist[t]
        w0, w1 = w0_acc, 1.0 - w0_acc
        if w0 == 0 or w1 == 0: continue
        mu0, mu1 = sum0_acc / w0, (total_mean - sum0_acc) / w1
        between = w0 * w1 * (mu0 - mu1) ** 2
        if between > max_between:
            max_between = between;
            best_thresh = t
    return (image < best_thresh).astype(np.uint8) * 255


def morphology_open_manual(img, ksize=3, iterations=1):
    out = img.copy()
    for _ in range(iterations):
        eroded = minimum_filter(out, size=ksize, mode='reflect')
        out = maximum_filter(eroded, size=ksize, mode='reflect')
    return out


def morphology_close_manual(img, ksize=3, iterations=1):
    out = img.copy()
    for _ in range(iterations):
        dilated = maximum_filter(out, size=ksize, mode='reflect')
        out = minimum_filter(dilated, size=ksize, mode='reflect')
    return out


def manual_blob_counter(binary_img, output_img, min_area, max_area, min_aspect, max_aspect):
    H, W = binary_img.shape
    binary_img_norm = binary_img.copy() // 255
    labels = np.zeros_like(binary_img_norm, dtype=np.int32)
    current_label = 1
    count = 0
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(H):
        for j in range(W):
            if binary_img_norm[i, j] == 1 and labels[i, j] == 0:
                stack = [(i, j)];
                ys, xs = [], [];
                labels[i, j] = current_label
                while stack:
                    y, x = stack.pop();
                    ys.append(y);
                    xs.append(x)
                    for dy, dx in dirs:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and binary_img_norm[ny, nx] == 1 and labels[ny, nx] == 0:
                            labels[ny, nx] = current_label;
                            stack.append((ny, nx))
                if not xs or not ys: continue
                x_min, x_max = min(xs), max(xs);
                y_min, y_max = min(ys), max(ys)
                w = (x_max - x_min + 1);
                h = (y_max - y_min + 1);
                area = w * h
                if w == 0 or h == 0: continue
                aspect_ratio = h / w
                is_person_shape = (aspect_ratio > min_aspect) and (aspect_ratio < max_aspect)
                if (min_area < area < max_area) and is_person_shape:
                    cv2.rectangle(output_img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)
                    count += 1
                current_label += 1
    return output_img, count


# --- Các hàm DÙNG OPENCV ---
def clahe_opencv(gray_img, clip_limit=2.0, tile_size=8):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray_img)


def remove_shadow_opencv(gray_img, k_size=21):
    """
    Loại bỏ ánh sáng/bóng không đồng đều bằng
    cách trừ đi một phiên bản mờ (Blur) của ảnh.
    Đây là một kỹ thuật Top-hat/Bottom-hat đơn giản.
    """
    if k_size % 2 == 0: k_size += 1  # Kernel phải lẻ
    blurred = cv2.GaussianBlur(gray_img, (k_size, k_size), 0)
    # Dùng cv2.subtract để trừ (an toàn, tránh giá trị âm)
    # Lấy ảnh gốc trừ ảnh mờ
    shadow_free = cv2.subtract(gray_img, blurred)
    # Nâng độ sáng tổng thể trở lại
    return cv2.add(shadow_free, 128)


def fixed_threshold_opencv(gray_img, thresh_val=127, invert=True):
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray_img, thresh_val, 255, mode)
    return binary


def otsu_threshold_opencv(gray_img, invert=True):
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray_img, 0, 255, mode + cv2.THRESH_OTSU)
    return binary


def adaptive_threshold_opencv(gray_img, block_size=21, C=5):
    if block_size % 2 == 0: block_size += 1
    return cv2.adaptiveThreshold(gray_img, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, block_size, C)


def morphology_opencv(binary_img, op_type="Close", k_size=5, iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    if op_type == "Close":
        op = cv2.MORPH_CLOSE
    elif op_type == "Open":
        op = cv2.MORPH_OPEN
    return cv2.morphologyEx(binary_img, op, kernel, iterations=iterations)


def contours_counter_opencv(binary_img, output_img, min_area, max_area, min_aspect, max_aspect):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for c in contours:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = h / w if w > 0 else 0
            if min_aspect < aspect_ratio < max_aspect:
                cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                count += 1
    return output_img, count


def hog_detector_opencv(original_img_rgb, scale=1.05, win_stride=4):
    hog = cv2.HOGDescriptor();
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, _ = hog.detectMultiScale(original_img_rgb,
                                    winStride=(win_stride, win_stride),
                                    padding=(8, 8),
                                    scale=scale)
    output_img = original_img_rgb.copy()
    for (x, y, w, h) in rects:
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
    return output_img, len(rects)


# --- GIAO DIỆN STREAMLIT ---

# --- THANH BÊN (SIDEBAR) ---
st.sidebar.title("Cấu hình Pipeline")
uploaded_file = st.sidebar.file_uploader("1. Tải ảnh lên", type=["jpg", "png", "jpeg"])

# --- CÁC BƯỚC (CHỈ HIỂN THỊ KHI ĐÃ TẢI ẢNH) ---
if uploaded_file:
    # --- Bước 1: Cân bằng Histogram ---
    with st.sidebar.expander("Bước 1: Cân bằng Histogram", expanded=True):
        method_enhance = st.selectbox("Phương pháp Cải thiện:", [
            "Không làm gì (Bỏ qua)",
            "Cân bằng Histogram (Thủ công)",
            "CLAHE (OpenCV)"
        ])
        # Tham số cho CLAHE
        if method_enhance == "CLAHE (OpenCV)":
            clip_limit = st.slider("Clip Limit", 1.0, 10.0, 2.0)
            tile_size = st.slider("Tile Size", 2, 16, 8)

    # --- Bước 2: Xóa bóng ---
    with st.sidebar.expander("Bước 2: Xóa bóng", expanded=True):
        method_shadow = st.selectbox("Phương pháp Xóa bóng:", [
            "Không làm gì (Bỏ qua)",
            "Xóa bóng (Thủ công"
            ")",
            "Xóa bóng (OpenCV)"
        ])
        # Tham số cho Xóa bóng (Thủ công)
        if method_shadow == "Xóa bóng (Thủ công)":
            k_shadow = st.slider("Kernel Size (Thủ công)", 5, 35, 15, 2, key="k_shadow_manual")
            s_shadow = st.slider("Sigma (Thủ công)", 1.0, 20.0, 5.0, key="s_shadow_manual")
        # Tham số cho Xóa bóng (OpenCV)
        elif method_shadow == "Xóa bóng (OpenCV)":
            k_shadow_cv = st.slider("Kernel Size (OpenCV)", 5, 51, 21, 2, key="k_shadow_cv")

    # --- Bước 3: Phân ngưỡng ---
    with st.sidebar.expander("Bước 3: Phân ngưỡng", expanded=True):
        method_thresh = st.selectbox("Phương pháp Phân ngưỡng:", [
            "Phân ngưỡng Thích nghi (OpenCV)",
            "Otsu (Thủ công)",
            "Otsu (OpenCV)",
            "Ngưỡng cố định (OpenCV)"
        ])
        # Tham số cho Thích nghi
        if method_thresh == "Phân ngưỡng Thích nghi (OpenCV)":
            block_size = st.slider("Block Size", 3, 51, 21, 2)
            C_val = st.slider("C Value", 1, 15, 5)
        # Tham số cho Ngưỡng cố định
        elif method_thresh == "Ngưỡng cố định (OpenCV)":
            thresh_val = st.slider("Ngưỡng", 0, 255, 127)

        # Tất cả các phương pháp trừ 'adaptive' đều có thể đảo ngược
        invert_thresh = st.checkbox("Đảo ngược (Vật thể tối -> Trắng)", value=True)

    # --- Bước 4: Xử lý hình thái ---
    with st.sidebar.expander("Bước 4: Xử lý hình thái", expanded=True):
        method_morph = st.selectbox("Phương pháp Hình thái:", [
            "Không làm gì (Bỏ qua)",
            "Đóng (Close) (OpenCV)",
            "Mở (Open) (OpenCV)",
            "Đóng -> Mở (OpenCV)",
            "Mở -> Đóng (Thủ công)"
        ])
        if method_morph != "Không làm gì (Bỏ qua)":
            k_morph = st.slider("Kernel Size", 3, 21, 5, 2)
            iter_morph = st.slider("Iterations", 1, 5, 1)

    # --- Bước 5: Đếm đối tượng ---
    with st.sidebar.expander("Bước 5: Đếm đối tượng", expanded=True):
        method_count = st.selectbox("Phương pháp Đếm:", [
            "Đếm Contours (OpenCV)",
            "Đếm Blob (Thủ công - DFS)",
            "Phát hiện HOG (OpenCV)"
        ])

        # HOG có tham số riêng
        if method_count == "Phát hiện HOG (OpenCV)":
            scale = st.slider("Hệ số tỷ lệ (Scale)", 1.01, 2.0, 1.05, 0.01)
            win_stride = st.select_slider("Bước nhảy (Stride)", [4, 8, 16], 8)
        # Các phương pháp đếm blob có tham số lọc
        else:
            st.write("Lọc đối tượng:")
            c1, c2 = st.columns(2)
            min_area = c1.number_input("Min Area", 100, 20000, 3000)
            max_area = c2.number_input("Max Area", 20000, 200000, 50000)
            min_aspect = c1.number_input("Min Aspect (H/W)", 0.1, 5.0, 1.5, 0.1)
            max_aspect = c2.number_input("Max Aspect (H/W)", 1.0, 10.0, 3.5, 0.1)

    # --- NÚT XỬ LÝ ---
    run_button = st.sidebar.button("Bắt đầu xử lý", type="primary", use_container_width=True)

else:
    # --- TRANG CHÍNH (KHI CHƯA TẢI ẢNH) ---
    st.title("Giới thiệu Ứng dụng Pipeline Xử lý ảnh")
    st.image("https://placehold.co/1200x400/20325A/FFFFFF?text=Pipeline+X%E1%BB%AD+L%C3%BD+%E1%BA%A2nh",
             use_container_width=True)
    st.info("Vui lòng tải một ảnh lên ở thanh bên trái để bắt đầu cấu hình pipeline.")
    run_button = False  # Không chạy gì cả

# --- VÒNG LẶP XỬ LÝ CHÍNH (KHI NHẤN NÚT) ---
if run_button:
    # --- Khởi tạo ảnh ---
    img_pil = Image.open(uploaded_file).convert('RGB')
    img_np_rgb = np.array(img_pil)

    st.title("Kết quả xử lý Pipeline")
    st.image(img_np_rgb, caption="Ảnh gốc", use_container_width=True)

    # --- CHUYỂN XÁM (Mặc định) ---
    st.header("Bước 0: Chuyển đổi Grayscale")
    with st.spinner("Đang chuyển sang ảnh xám..."):
        img_np_gray = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2GRAY)
        current_image = img_np_gray.copy()  # Đây là ảnh sẽ được xử lý
    st.image(current_image, caption="Kết quả: Ảnh xám", use_container_width=True, clamp=True)

    # --- Xử lý Bước 4 (HOG) - Chạy riêng ---
    # Nếu chọn HOG, bỏ qua tất cả các bước khác và chạy HOG trên ảnh RGB gốc
    if method_count == "Phát hiện HOG (OpenCV)":
        st.header("Bước 1, 2, 3: Bỏ qua (HOG chạy trên ảnh gốc)")
        st.header("Bước 4: Đếm đối tượng")
        st.warning("Phương pháp HOG chạy trực tiếp trên ảnh RGB gốc, bỏ qua các bước xử lý ảnh xám.")

        with st.spinner(f"Đang áp dụng: {method_count}..."):
            final_image, count = hog_detector_opencv(img_np_rgb, scale, win_stride)

        st.success(f"Hoàn thành! Đã phát hiện tổng cộng: {count} đối tượng")
        st.image(final_image, caption=f"Kết quả cuối cùng (HOG): {count} đối tượng", use_container_width=True)

    # --- Xử lý Pipeline thông thường (cho các phương pháp đếm blob) ---
    else:
        # --- Xử lý Bước 1: Cân bằng Histogram ---
        st.header("Bước 1: Cân bằng Histogram")
        with st.spinner(f"Đang áp dụng: {method_enhance}..."):
            if method_enhance == "Cân bằng Histogram (Thủ công)":
                current_image = hist_equalize_manual(current_image)
            elif method_enhance == "CLAHE (OpenCV)":
                current_image = clahe_opencv(current_image, clip_limit, tile_size)
            # else: "Không làm gì" -> current_image giữ nguyên
        st.image(current_image, caption=f"Kết quả Bước 1: {method_enhance}", use_container_width=True, clamp=True)

        # --- Xử lý Bước 2: Xóa bóng ---
        st.header("Bước 2: Xóa bóng")
        with st.spinner(f"Đang áp dụng: {method_shadow}..."):
            if method_shadow == "Xóa bóng (Thủ công)":
                current_image = remove_shadow_manual(current_image, k_shadow, s_shadow)
            elif method_shadow == "Xóa bóng (OpenCV)":
                current_image = remove_shadow_opencv(current_image, k_shadow_cv)
            # else: "Không làm gì" -> current_image giữ nguyên
        st.image(current_image, caption=f"Kết quả Bước 2: {method_shadow}", use_container_width=True, clamp=True)

        # --- Xử lý Bước 3: Phân ngưỡng ---
        st.header("Bước 3: Phân ngưỡng")
        with st.spinner(f"Đang áp dụng: {method_thresh}..."):
            if method_thresh == "Phân ngưỡng Thích nghi (OpenCV)":
                # Adaptive threshold yêu cầu ảnh đầu vào là 8-bit
                current_image_thresh = adaptive_threshold_opencv(current_image, block_size, C_val)
            elif method_thresh == "Otsu (Thủ công)":
                current_image_thresh = global_hist_threshold_manual(current_image)
            elif method_thresh == "Otsu (OpenCV)":
                current_image_thresh = otsu_threshold_opencv(current_image, invert_thresh)
            elif method_thresh == "Ngưỡng cố định (OpenCV)":
                current_image_thresh = fixed_threshold_opencv(current_image, thresh_val, invert_thresh)
        st.image(current_image_thresh, caption=f"Kết quả Bước 3: {method_thresh}", use_container_width=True, clamp=True)

        # --- Xử lý Bước 4: Hình thái ---
        st.header("Bước 4: Xử lý hình thái")
        with st.spinner(f"Đang áp dụng: {method_morph}..."):
            if method_morph == "Mở -> Đóng (Thủ công)":
                opened = morphology_open_manual(current_image_thresh, k_morph, iter_morph)
                current_image_morphed = morphology_close_manual(opened, k_morph, iter_morph)
            elif method_morph == "Đóng (Close) (OpenCV)":
                current_image_morphed = morphology_opencv(current_image_thresh, "Close", k_morph, iter_morph)
            elif method_morph == "Mở (Open) (OpenCV)":
                current_image_morphed = morphology_opencv(current_image_thresh, "Open", k_morph, iter_morph)
            elif method_morph == "Đóng -> Mở (OpenCV)":
                closed = morphology_opencv(current_image_thresh, "Close", k_morph, iter_morph)
                current_image_morphed = morphology_opencv(closed, "Open", k_morph, iter_morph)
            else:  # "Không làm gì"
                current_image_morphed = current_image_thresh.copy()
        st.image(current_image_morphed, caption=f"Kết quả Bước 4: {method_morph}", use_container_width=True, clamp=True)

        # --- Xử lý Bước 5: Đếm ---
        st.header("Bước 5: Đếm đối tượng")
        # Tạo ảnh nền để vẽ (từ ảnh gốc)
        output_display_img = img_np_rgb.copy()

        with st.spinner(f"Đang áp dụng: {method_count}..."):
            if method_count == "Đếm Contours (OpenCV)":
                final_image, count = contours_counter_opencv(
                    current_image_morphed, output_display_img,
                    min_area, max_area, min_aspect, max_aspect
                )
            elif method_count == "Đếm Blob (Thủ công - DFS)":
                final_image, count = manual_blob_counter(
                    current_image_morphed, output_display_img,
                    min_area, max_area, min_aspect, max_aspect
                )

        st.success(f"Hoàn thành! Đã phát hiện tổng cộng: {count} đối tượng")
        st.image(final_image, caption=f"Kết quả cuối cùng ({method_count}): {count} đối tượng",
                 use_container_width=True)
