from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = 'secret_key'

# 사용자 데이터에 name 추가
users = {
    'admin': {
        'password': '1234',
        'balance': 1000,
        'name': 'user1'  # 실제 보여줄 이름
    }
}

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        if 'username' in session:  # 입금/출금 처리
            username = session['username']
            user = users[username]
            message = ""

            try:
                amount = int(request.form['amount'])
                action = request.form['action']
                if action == '입금':
                    user['balance'] += amount
                    message = f"{amount}원 입금되었습니다."
                elif action == '출금':
                    if user['balance'] >= amount:
                        user['balance'] -= amount
                        message = f"{amount}원 출금되었습니다."
                    else:
                        message = "잔액이 부족합니다."
            except:
                message = "금액을 숫자로 입력하세요."

            return render_template("main.html",
                                   mode="dashboard",
                                   display_name=user['name'],
                                   balance=user['balance'],
                                   message=message)

        else:  # 로그인 시도
            username = request.form['username']
            password = request.form['password']
            user = users.get(username)

            if user and user['password'] == password:
                session['username'] = username
                return redirect(url_for('main'))
            else:
                return render_template("main.html", mode="login", message="로그인 실패: 아이디 또는 비밀번호가 틀렸습니다.")

    if 'username' in session:
        username = session['username']
        user = users[username]
        return render_template("main.html",
                               mode="dashboard",
                               display_name=user['name'],
                               balance=user['balance'])
    else:
        return render_template("main.html", mode="login")


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('main'))

if __name__ == '__main__':
    app.run(debug=True)
