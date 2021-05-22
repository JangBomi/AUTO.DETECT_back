from django.contrib import admin

# Register your models here.

from .models import Post
from .models import User
from .models import Record
from .models import RecordDetail

admin.site.register(Post)
admin.site.register(User)
admin.site.register(Record)
admin.site.register(RecordDetail)
