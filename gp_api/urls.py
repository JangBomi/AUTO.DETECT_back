
from django.urls import path
from django.conf.urls import url
from .views import NoteViewSet, RegistrationAPI, UserAPI

from . import views


note_list = NoteViewSet.as_view({"get": "list", "post": "create"})

note_detail = NoteViewSet.as_view(
    {"get": "retrieve", "patch": "partial_update", "delete": "destroy"}
)


urlpatterns = [
    path('', views.ListPost.as_view()),
    path('<int:pk>/', views.DetailPost.as_view()),
    url("^notes/$", note_list, name="note-list"),
    url("^notes/(?P<pk>[0-9]+)/$", note_detail, name="note-detail"),
    url("^auth/register/$", RegistrationAPI.as_view()),
    url("^auth/user/$", UserAPI.as_view()),
    path('signup', views.createUser),
    path('signin', views.login),
    path('home/record', views.record),
    path('home/record/detail/<int:recordId_id>', views.recordDetail)
]

