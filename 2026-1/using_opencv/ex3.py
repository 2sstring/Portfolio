import cv2

# 웹캠 연결 (0은 기본 웹캠)
cap = cv2.VideoCapture(0)

# 웹캠이 성공적으로 열렸는지 확인
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
else:
    # 웹캠이 열려있는 동안 계속 프레임 읽기
    while cap.isOpened():
        # 프레임 읽기
        ret, frame = cap.read()
        
        # 프레임이 제대로 읽혔는지 확인
        if not ret:
            print("프레임 읽기 오류")
            break
        
        # 프레임 표시
        cv2.imshow('Webcam', frame)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()
