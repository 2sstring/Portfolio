import cv2
import numpy as np

# desert.JPG 이미지 읽기
image = cv2.imread('desert.JPG')

# 이미지가 성공적으로 읽혔는지 확인
if image is None:
    print("desert.JPG 파일을 찾을 수 없습니다.")
else:
    # 원본 이미지 표시
    cv2.imshow('Original Image', image)
    
    # 그레이스케일로 변환
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale Image', gray_image)
    
    # 가우시안 블러로 노이즈 제거
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    cv2.imshow('Blurred Image', blurred_image)
    
    # filter2D를 사용한 엣지 검출 커널 정의
    
    # Sobel X 커널 (수직 엣지 검출)
    sobel_x_kernel = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], dtype=np.float32)
    
    # Sobel Y 커널 (수평 엣지 검출)
    sobel_y_kernel = np.array([[-1, -2, -1],
                             [ 0,  0,  0],
                             [ 1,  2,  1]], dtype=np.float32)
    
    # Laplacian 커널 (모든 방향의 엣지 검출)
    laplacian_kernel = np.array([[ 0, -1,  0],
                               [-1,  4, -1],
                               [ 0, -1,  0]], dtype=np.float32)
    
    # Prewitt X 커널
    prewitt_x_kernel = np.array([[-1, 0, 1],
                                [-1, 0, 1],
                                [-1, 0, 1]], dtype=np.float32)
    
    # Prewitt Y 커널
    prewitt_y_kernel = np.array([[-1, -1, -1],
                                [ 0,  0,  0],
                                [ 1,  1,  1]], dtype=np.float32)
    
    # Roberts 커널
    roberts_x_kernel = np.array([[1, 0],
                               [0, -1]], dtype=np.float32)
    
    roberts_y_kernel = np.array([[0, 1],
                               [-1, 0]], dtype=np.float32)
    
    # filter2D를 사용한 엣지 검출
    sobel_x_filter = cv2.filter2D(blurred_image, cv2.CV_64F, sobel_x_kernel)
    sobel_y_filter = cv2.filter2D(blurred_image, cv2.CV_64F, sobel_y_kernel)
    
    # 결과를 절대값으로 변환하고 8비트로 변환
    sobel_x_filter = np.absolute(sobel_x_filter)
    sobel_y_filter = np.absolute(sobel_y_filter)
    sobel_x_filter = np.uint8(sobel_x_filter)
    sobel_y_filter = np.uint8(sobel_y_filter)
    
    cv2.imshow('Sobel X (filter2D)', sobel_x_filter)
    cv2.imshow('Sobel Y (filter2D)', sobel_y_filter)
    
    # Sobel X와 Y 결합
    sobel_combined_filter = cv2.bitwise_or(sobel_x_filter, sobel_y_filter)
    cv2.imshow('Sobel Combined (filter2D)', sobel_combined_filter)
    
    # Laplacian 필터
    laplacian_filter = cv2.filter2D(blurred_image, cv2.CV_64F, laplacian_kernel)
    laplacian_filter = np.absolute(laplacian_filter)
    laplacian_filter = np.uint8(laplacian_filter)
    cv2.imshow('Laplacian (filter2D)', laplacian_filter)
    
    # Prewitt 필터
    prewitt_x_filter = cv2.filter2D(blurred_image, cv2.CV_64F, prewitt_x_kernel)
    prewitt_y_filter = cv2.filter2D(blurred_image, cv2.CV_64F, prewitt_y_kernel)
    
    prewitt_x_filter = np.absolute(prewitt_x_filter)
    prewitt_y_filter = np.absolute(prewitt_y_filter)
    prewitt_x_filter = np.uint8(prewitt_x_filter)
    prewitt_y_filter = np.uint8(prewitt_y_filter)
    
    prewitt_combined = cv2.bitwise_or(prewitt_x_filter, prewitt_y_filter)
    cv2.imshow('Prewitt Combined (filter2D)', prewitt_combined)
    
    # Roberts 필터
    roberts_x_filter = cv2.filter2D(blurred_image, cv2.CV_64F, roberts_x_kernel)
    roberts_y_filter = cv2.filter2D(blurred_image, cv2.CV_64F, roberts_y_kernel)
    
    roberts_x_filter = np.absolute(roberts_x_filter)
    roberts_y_filter = np.absolute(roberts_y_filter)
    roberts_x_filter = np.uint8(roberts_x_filter)
    roberts_y_filter = np.uint8(roberts_y_filter)
    
    roberts_combined = cv2.bitwise_or(roberts_x_filter, roberts_y_filter)
    cv2.imshow('Roberts Combined (filter2D)', roberts_combined)
    
    # 사용자 정의 엣지 검출 커널
    custom_kernel = np.array([[-1, -1, -1],
                             [-1,  8, -1],
                             [-1, -1, -1]], dtype=np.float32)
    
    custom_edges = cv2.filter2D(blurred_image, cv2.CV_64F, custom_kernel)
    custom_edges = np.absolute(custom_edges)
    custom_edges = np.uint8(custom_edges)
    cv2.imshow('Custom Edge Detection', custom_edges)
    
    # 원본 이미지에 엣지 겹쳐서 표시
    edges_colored = cv2.cvtColor(sobel_combined_filter, cv2.COLOR_GRAY2BGR)
    combined = cv2.addWeighted(image, 0.7, edges_colored, 0.3, 0)
    cv2.imshow('Original with Edges', combined)
    
    # 결과 이미지 저장
    cv2.imwrite('sobel_x_filter2d.png', sobel_x_filter)
    cv2.imwrite('sobel_y_filter2d.png', sobel_y_filter)
    cv2.imwrite('sobel_combined_filter2d.png', sobel_combined_filter)
    cv2.imwrite('laplacian_filter2d.png', laplacian_filter)
    cv2.imwrite('prewitt_combined_filter2d.png', prewitt_combined)
    cv2.imwrite('roberts_combined_filter2d.png', roberts_combined)
    cv2.imwrite('custom_edges_filter2d.png', custom_edges)
    cv2.imwrite('original_with_edges_filter2d.png', combined)
    
    print("filter2D 엣지 검출 결과를 저장했습니다:")
    print("- sobel_x_filter2d.png")
    print("- sobel_y_filter2d.png")
    print("- sobel_combined_filter2d.png")
    print("- laplacian_filter2d.png")
    print("- prewitt_combined_filter2d.png")
    print("- roberts_combined_filter2d.png")
    print("- custom_edges_filter2d.png")
    print("- original_with_edges_filter2d.png")
    
    # 이미지 정보 출력
    print(f"\n이미지 크기: {image.shape[1]}x{image.shape[0]}")
    print(f"그레이스케일 크기: {gray_image.shape[1]}x{gray_image.shape[0]}")
    
    # 키 입력 대기 (아무 키나 누르면 창이 닫힘)
    cv2.waitKey(0)
    
    # 모든 창 닫기
    cv2.destroyAllWindows()
