import cv2

# 비디오 파일 읽기
cap = cv2.VideoCapture('test_video.mp4')

# 비디오 파일이 성공적으로 열렸는지 확인
if not cap.isOpened():
    print("test_video.mp4 파일을 열 수 없습니다.")
else:
    # 비디오가 끝날 때까지 프레임 읽기
    while cap.isOpened():
        # 프레임 읽기
        ret, frame = cap.read()
        
        # 프레임이 제대로 읽혔는지 확인
        if not ret:
            print("비디오 끝 또는 읽기 오류")
            break
        
        # 프레임 표시
        cv2.imshow('Video', frame)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()
