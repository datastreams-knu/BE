from flask import Flask, redirect, request, session
import requests

app = Flask(__name__)
app.secret_key = 'your_secret_key'

client_id = 'your_kakao_client_id'
#redirect url 은 로그인 창으로 
redirect_uri = 'login window url'

@app.route('/login')
def login():
    kakao_auth_url = f"https://kauth.kakao.com/oauth/authorize?client_id={client_id}&redirect_uri={redirect_uri}&response_type=code"
    return redirect(kakao_auth_url)

@app.route('/callback')
def callback():
    code = request.args.get('code')
    token_url = 'https://kauth.kakao.com/oauth/token'
    token_data = {
        'grant_type': 'authorization_code',
        'client_id': client_id,
        'redirect_uri': redirect_uri,
        'code': code
    }
    token_response = requests.post(token_url, data=token_data)
    access_token = token_response.json().get('access_token')
    
    user_info_url = 'https://kapi.kakao.com/v2/user/me'
    headers = {'Authorization': f'Bearer {access_token}'}
    user_info_response = requests.get(user_info_url, headers=headers)
    user_info = user_info_response.json()
    
    kakao_id = user_info.get('id')
    session['user_id'] = kakao_id  # user id 를 세션에 저장.

    return '로그인 성공'

if __name__ == '__main__':
    app.run()
