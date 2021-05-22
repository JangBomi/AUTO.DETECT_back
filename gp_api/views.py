from django.shortcuts import render
from rest_framework import generics, viewsets, permissions, status
from rest_framework.response import Response
from knox.models import AuthToken
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from rest_framework.permissions import AllowAny
from rest_framework.permissions import IsAuthenticated
from rest_framework_jwt.authentication import JSONWebTokenAuthentication


from .models import Post, Notes, User
from .serializers import (
    PostSerializer,
    NoteSerializer,
    CreateUserSerializer,
    UserSerializer,
    LoginSerializer,
    RecordSerializer,
    RecordDetailSerializer
)


class ListPost(generics.ListCreateAPIView):
    queryset = Post.objects.all()
    serializer_class = PostSerializer


class DetailPost(generics.RetrieveUpdateDestroyAPIView):
    queryset = Post.objects.all()
    serializer_class = PostSerializer


class NoteViewSet(viewsets.ModelViewSet):
    permission_classes = [permissions.IsAuthenticated, ]
    serializer_class = NoteSerializer

    def get_queryset(self):
        return self.request.user.notes.all().order_by("-created_at")

    def perform_create(self, serializer):
        serializer.save(owner=self.request.user)


@api_view(['POST'])
@permission_classes([AllowAny])
def createUser(request):
    if request.method == 'POST':
        serializer = CreateUserSerializer(data=request.data)
        if not serializer.is_valid(raise_exception=True):
            return Response({"message": "Request Body Error."}, status=status.HTTP_409_CONFLICT)

        if User.objects.filter(email=serializer.validated_data['email']).first() is None:
            serializer.save()
            return Response({"message": "ok"}, status=status.HTTP_201_CREATED)
        return Response({"message": "duplicate email"}, status=status.HTTP_409_CONFLICT)


@api_view(['POST', 'GET'])
@permission_classes([AllowAny])
def login(request):
    if request.method == 'POST':
        serializer = LoginSerializer(data=request.data)

        if not serializer.is_valid(raise_exception=True):
            return Response({"message": "Request Body Error."}, status=status.HTTP_409_CONFLICT)
        if serializer.validated_data['email'] == "None":
            return Response({'message': 'fail'}, status=status.HTTP_200_OK)

        response = {
            'success': 'True',
            'token': serializer.data['token']
        }
        return Response(response, status=status.HTTP_200_OK)


# data=request.query_param


@api_view(['POST', 'GET'])
@permission_classes((IsAuthenticated, ))
@authentication_classes((JSONWebTokenAuthentication,))
def record(request):
    if request.method == 'GET':
        serializer = RecordSerializer()

        response = {
            'success': True,
            'message': "successfully get record",
            'data': serializer.get_all()
        }

        return Response(response, status=status.HTTP_200_OK)

    if request.method == 'POST':
        serializer = RecordSerializer(data=request.data)

        if not serializer.is_valid(raise_exception=True):
            return Response({"success": False, "message": "Request Body Error."}, status=status.HTTP_409_CONFLICT)

        serializer.save()

        response = {
            'success': True,
            'message': "successfully save record"
        }

        return Response(response, status=status.HTTP_201_CREATED)


@api_view(['POST', 'GET'])
@permission_classes([AllowAny])
def recordDetail(request, recordId_id):
    if request.method == 'GET':
        serializer = RecordDetailSerializer()

        response = {
            'success': True,
            'message': "successfully get record",
            'data': serializer.get(recordId_id)
        }

        return Response(response, status=status.HTTP_200_OK)

        # if recordId_id is None:
    if request.method == 'POST':
        serializer = RecordDetailSerializer(data=request.data)

        if not serializer.is_valid(raise_exception=True):
            return Response({"success": False, "message": "Error."}, status=status.HTTP_409_CONFLICT)

        #serializer.create(validated_data=request.data)
        serializer.save()

        response = {
            'success': True,
            'message': "successfully save record detail"
        }

        return Response(response, status=status.HTTP_201_CREATED)


class RegistrationAPI(generics.GenericAPIView):
    serializer_class = CreateUserSerializer

    def post(self, request, *args, **kwargs):
        if len(request.data["username"]) < 6 or len(request.data["password"]) < 4:
            body = {"message": "short field"}
            return Response(body, status=status.HTTP_400_BAD_REQUEST)
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        return Response(
            {
                "user": UserSerializer(
                    user, context=self.get_serializer_context()
                ).data,
                "token": AuthToken.objects.create(user)[1],
            }
        )


# class LoginAPI(generics.GenericAPIView):
#     serializer_class = LoginUserSerializer
#
#     def post(self, request, *args, **kwargs):
#         serializer = self.get_serializer(data=request.data)
#         serializer.is_valid(raise_exception=True)
#         user = serializer.validated_data
#         return Response(
#             {
#                 "user": UserSerializer(
#                     user, context=self.get_serializer_context()
#                 ).data,
#                 "token": AuthToken.objects.create(user)[1],
#             }
#         )


class UserAPI(generics.RetrieveAPIView):
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = UserSerializer

    def get_object(self):
        return self.request.user
