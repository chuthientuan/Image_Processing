import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter


# --- TẤT CẢ CÁC HÀM XỬ LÝ ẢNH CỦA BẠN (giữ nguyên) ---

def remove_shadow(gray_img, kernel_size=15, sigma=5, eps=1e-3):
    """
    Hàm tự viết để xóa bóng (từ code của bạn).
    Sử dụng phép chia ảnh cho phiên bản mờ của nó.
    """
    k = kernel_size // 2
    x, y = np.mgrid[-k:k + 1, -k:k + 1]
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()

    H, W = gray_img.shape
    padded = np.pad(gray_img, ((k, k), (k, k)), mode='reflect')
    blurred = np.zeros_like(gray_img, dtype=np.float32)
    # Tối ưu hóa: Thay vì 2 vòng lặp 'for', chúng ta có thể dùng
    # hàm 'filter2D' của OpenCV hoặc 'convolve2d' của Scipy
    # Nhưng ta sẽ giữ nguyên logic của bạn để đảm bảo kết quả
    for i in range(kernel_size):
        for j in range(kernel_size):
            blurred += kernel[i, j] * padded[i:i + H, j:j + W]

    shadow_free = gray_img.astype(np.float32) / (blurred.astype(np.float32) / 255.0 + eps)
    shadow_free = np.clip(shadow_free, 0, 255).astype(np.uint8)
    return shadow_free


def global_hist_threshold(image):
    """
    Hàm tự viết để tìm ngưỡng Otsu toàn cục.
    """
    hist = np.zeros(256, dtype=np.float32)
    for val in image.ravel():
        hist[val] += 1
    hist /= hist.sum()

    bins = np.arange(256)
    total_mean = np.sum(bins * hist)
    best_thresh = 0
    max_between = -1
    w0_acc, sum0_acc = 0.0, 0.0

    for t in range(256):
        w0_acc += hist[t]
        sum0_acc += t * hist[t]
        w0 = w0_acc
        w1 = 1.0 - w0
        if w0 == 0 or w1 == 0: continue
        mu0 = sum0_acc / w0
        mu1 = (total_mean - sum0_acc) / w1
        between = w0 * w1 * (mu0 - mu1) ** 2
        if between > max_between:
            max_between = between
            best_thresh = t

    # Logic của bạn: (image < best_thresh) -> người là màu trắng
    # Điều này giả định nền sáng hơn người.
    binary = (image < best_thresh).astype(np.uint8) * 255
    return binary


def morphology_open(img, ksize=2, iterations=1):
    """
    Hàm Mở (Opening).
    """
    out = img.copy()
    for _ in range(iterations):
        eroded = minimum_filter(out, size=ksize, mode='reflect')
        out = maximum_filter(eroded, size=ksize, mode='reflect')
    return out


def morphology_close(img, ksize=2, iterations=1):
    """
    Hàm Đóng (Closing).
    """
    out = img.copy()
    for _ in range(iterations):
        dilated = maximum_filter(out, size=ksize, mode='reflect')
        out = minimum_filter(dilated, size=ksize, mode='reflect')
    return out


# --- KẾT THÚC CÁC HÀM XỬ LÝ ẢNH ---


# --- LOGIC ỨNG DỤNG STREAMLIT ---

st.set_page_config(layout="wide")
st.title("Ứng dụng Xử lý ảnh - Đếm đối tượng trong ảnh camera giám sát")

# 1. Tải ảnh lên
uploaded_file = st.file_uploader("Chọn một file ảnh...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Đọc ảnh
    img = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(img)  # H x W x 3
    H, W = img_np.shape[:2]

    # SỬA LỖI: Đã thay use_column_width bằng use_container_width
    st.image(img_np, caption="Ảnh gốc", use_container_width=True)

    # Bắt đầu xử lý
    with st.spinner("Đang xử lý ảnh... (các hàm tự viết có thể hơi chậm)"):

        # --- 2. Chuyển xám ---
        st.header("Bước 1: Chuyển đổi Grayscale")
        gray = (0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2]).astype(np.uint8)
        # SỬA LỖI: Đã thay use_column_width bằng use_container_width
        st.image(gray, caption="Ảnh xám (Grayscale)", use_container_width=True)

        # --- 3. Cân bằng Histogram ---
        st.header("Bước 2: Cân bằng Histogram")
        hist = np.zeros(256, dtype=int)
        for v in gray.flatten():
            hist[v] += 1

        cdf = np.cumsum(hist)
        cdf_min = cdf[cdf > 0][0]
        total_pixels = gray.size
        lut = ((cdf - cdf_min) * 255 / (total_pixels - cdf_min)).clip(0, 255).astype(np.uint8)
        img_enhanced = lut[gray]
        gray_eq = img_enhanced.copy()
        # SỬA LỖI: Đã thay use_column_width bằng use_container_width
        st.image(img_enhanced, caption="Ảnh sau khi cân bằng Histogram", use_container_width=True)

        # --- 4. Xóa bóng ---
        st.header("Bước 3: Xóa bóng")
        shadow_free = remove_shadow(gray_eq, kernel_size=15, sigma=5)
        # SỬA LỖI: Đã thay use_column_width bằng use_container_width
        st.image(shadow_free, caption="Ảnh sau khi xóa bóng", use_container_width=True)

        st.header("Bước 4: Phân ngưỡng Otsu")
        segmented_otsu = global_hist_threshold(shadow_free)
        st.image(segmented_otsu, caption="Ảnh nhị phân (Otsu)", use_container_width=True)

        st.header("Bước 5: Xử lý hình thái (Mở -> Đóng)")
        opened = morphology_open(segmented_otsu, ksize=3, iterations=1)
        closed = morphology_close(opened, ksize=3, iterations=1)

        col1, col2 = st.columns(2)
        col1.image(opened, caption="Ảnh sau khi Mở (Loại bỏ nhiễu)", use_container_width=True)
        col2.image(closed, caption="Ảnh sau khi Đóng (Lấp đầy lỗ hổng)", use_container_width=True)

        st.header("Bước 6: Tìm và Đếm đối tượng")
        H, W = closed.shape
        binary_img = closed.copy() // 255

        output = np.stack([img_enhanced] * 3, axis=-1)

        min_area, max_area = 3000, 50000
        min_aspect_ratio = 1.5
        max_aspect_ratio = 3.5

        count = 0
        labels = np.zeros_like(binary_img, dtype=np.int32)
        current_label = 1
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for i in range(H):
            for j in range(W):
                if binary_img[i, j] == 1 and labels[i, j] == 0:
                    stack = [(i, j)]
                    ys, xs = [], []
                    labels[i, j] = current_label
                    while stack:
                        y, x = stack.pop()
                        ys.append(y)
                        xs.append(x)
                        for dy, dx in dirs:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < H and 0 <= nx < W:
                                if binary_img[ny, nx] == 1 and labels[ny, nx] == 0:
                                    labels[ny, nx] = current_label
                                    stack.append((ny, nx))

                    if not xs or not ys: continue

                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)

                    w = (x_max - x_min + 1)
                    h = (y_max - y_min + 1)
                    area = w * h

                    if w == 0 or h == 0: continue
                    aspect_ratio = h / w

                    is_person_shape = (aspect_ratio > min_aspect_ratio) and (aspect_ratio < max_aspect_ratio)

                    if (min_area < area < max_area) and is_person_shape:
                        # Vẽ hộp (theo logic của bạn)
                        output[y_min:y_min + 2, x_min:x_max + 1] = [255, 0, 0]  # top
                        output[y_max:y_max + 1, x_min:x_max + 1] = [255, 0, 0]  # bottom
                        output[y_min:y_max + 1, x_min:x_min + 2] = [255, 0, 0]  # left
                        output[y_min:y_max + 1, x_max:x_max + 1] = [255, 0, 0]  # right
                        count += 1
                    current_label += 1

        st.success(f"Hoàn thành! Đã phát hiện tổng cộng: {count} đối tượng")
        st.image(output, caption=f"Kết quả cuối cùng: {count} đối tượng", use_container_width=True)

else:
    st.info("Vui lòng tải một ảnh lên để bắt đầu xử lý.")
