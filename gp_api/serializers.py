#from django.contrib.auth.models import User
from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import Post, Notes, User, Record, RecordDetail
from rest_framework_jwt.settings import api_settings
from django.contrib.auth.models import update_last_login
from django.contrib.auth import authenticate

JWT_PAYLOAD_HANDLER = api_settings.JWT_PAYLOAD_HANDLER
JWT_ENCODE_HANDLER = api_settings.JWT_ENCODE_HANDLER

User = get_user_model()


class CreateUserSerializer(serializers.Serializer):
    email = serializers.EmailField(required=True)
    username = serializers.CharField(required=True)
    password = serializers.CharField(required=True)
    # repassword = serializers.CharField(required=True) #비밀번호 재확인 위해
    realUserName = serializers.CharField(required=True)
    birthDate = serializers.DateField(required=True)

    def create(self, validated_data):
        user = User.objects.create(
            email=validated_data['email'],
            username=validated_data['username'],
            realUserName=validated_data['realUserName'],
            birthDate=validated_data['birthDate'],
        )
        user.set_password(validated_data['password'])

        user.save()
        return user


class LoginSerializer(serializers.Serializer):
    email = serializers.CharField(max_length=64)
    password = serializers.CharField(max_length=128, write_only=True)
    token = serializers.CharField(max_length=255, read_only=True)

    def validate(self, data):
        email = data.get("email", None)
        password = data.get("password", None)
        user = authenticate(email=email, password=password)

        if user is None:
            return {
                'email': 'None'
            }
        try:
            payload = JWT_PAYLOAD_HANDLER(user)
            jwt_token = JWT_ENCODE_HANDLER(payload)
            update_last_login(None, user)
        except User.DoesNotExist:
            raise serializers.ValidationError(
                'User with given email and password does not exists'
            )

        return_value = {
            'id': user.id,
            'email': user.email,
            'token': jwt_token,
        }
        return return_value


class RecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = Record
        fields = (
            'id',
            'title',
            'startTime',
            'endTime',
            'recordNum',
            'etc',
            'userId'
        )

    title = serializers.CharField(max_length=255)
    startTime = serializers.DateTimeField()
    endTime = serializers.DateTimeField
    recordNum = serializers.IntegerField()
    etc = serializers.CharField(max_length=255)
    userId = serializers.IntegerField

    def create(self, validated_data):
        record = Record.objects.create(
            title=validated_data['title'],
            startTime=validated_data['startTime'],
            endTime=validated_data['endTime'],
            recordNum=validated_data['recordNum'],
            etc=validated_data['etc'],
            userId=validated_data['userId']
        )

        record.save()
        return record

    def get_all(self, userId_id):
        print(userId_id)
        # print(request)
        # print(request.data)
        # userId = request.GET['userId']
        queryset = Record.objects.all().filter(userId_id=userId_id).order_by('-id')
        record = RecordSerializer(queryset, many=True)

        return record.data


class RecordUpdateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Record
        fields = (
            'id',
            'endTime',
            'recordNum',
            'etc',
            'userId'
        )

    endTime = serializers.DateTimeField
    recordNum = serializers.IntegerField()
    etc = serializers.CharField(max_length=255)
    userId = serializers.IntegerField

    def updateRecord(self, validated_data, record_id):
        count = 0
        recordDetail = RecordDetail.objects.filter(recordId_id=record_id)

        for detail in recordDetail:
            count = count + 1

        presentRecord = Record.objects.filter(id=record_id)

        for record in presentRecord:
            record.endTime = validated_data['endTime']
            record.etc = validated_data['etc']
            record.recordNum = count
            record.save()

        return presentRecord

    def deleteRecord(self, validated_data, record_id):

        record = Record.objects.filter(id=record_id)
        for r in record:
            r.delete()


class RecordDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = RecordDetail
        fields = (
            'id',
            'detectedItem',
            'image',
            'captureTime',
            'recordId_id'
        )

    detectedItem = serializers.CharField()
    image = serializers.CharField()
    captureTime = serializers.DateTimeField()
    recordId_id = serializers.IntegerField()

    def create(self, validated_data):
        recordDetail = RecordDetail.objects.create(
            detectedItem=validated_data['detectedItem'],
            image="https://gpbucket-bomi.s3.ap-northeast-2.amazonaws.com/"+validated_data['recordId_id']+"/"+validated_data['image'],
            captureTime=validated_data['captureTime'],
            recordId_id=validated_data['recordId_id']
        )

        recordDetail.save()
        return recordDetail

    def get(self, recordId_id):
        queryset = RecordDetail.objects.all().filter(recordId_id=recordId_id)
        recordDetail = RecordDetailSerializer(queryset, many=True)

        return recordDetail.data


class PostSerializer(serializers.ModelSerializer):
    class Meta:
        fields = (
            'id',
            'title',
            'content',
        )
        model = Post


class NoteSerializer(serializers.ModelSerializer):
    class Meta:
        model = Notes
        fields = ("id", "text")

# 회원가입 시리얼라이저


# class CreateUserSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = User
#         fields = ("id", "username", "password")
#         extra_kwargs = {"password": {"write_only": True}}
#
#     def create(self, validated_data):
#         user = User.objects.create_user(
#             validated_data["username"], None, validated_data["password"]
#         )
#         return user


# 접속 유지중인지 확인할 시리얼라이저

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ("id", "username")


#blogs = Blog.objects.filter(author=author).values_list('id', flat=True)

# 로그인 시리얼라이저

# class LoginUserSerializer(serializers.Serializer):
#     username = serializers.CharField()
#     password = serializers.CharField()
#
#     def validate(self, data):
#         user = authenticate(**data)
#         if user and user.is_active:
#             return user
#         raise serializers.ValidationError("Unable to log in with provided credentials.")
