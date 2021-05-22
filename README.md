# 패키지 다운
pip install -r requirements.txt
<br>

## requirements.txt 생성
pip freeze > requirements.txt

# 시작
python manage.py runserver

# 마이그레이션 파일 생성
python manage.py makemigrations <app-name>

# 마이그레이션 적용
python manage.py migrate <app-name>

# 마이그레이션 적용 현황 
python manage.py showmigrations <app-name>

# 지정 마이그레이션의 SQL 내역
python manage.py sqlmigrate <app-name> <migration-name>

# db 실행
./manage.py dbshell