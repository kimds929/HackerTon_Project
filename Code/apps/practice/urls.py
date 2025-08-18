from django.urls import path
from . import views

app_name = "practice"

urlpatterns = [
    path("", views.practice_main, name="practice_main"),
    path("attempt/", views.practice_attempt, name="practice_attempt"),
    path("retrieve_set/", views.retrieve_practice_set, name="retrieve_practice_set"),
    path("submit_answer/", views.submit_answer, name="submit_answer"),
    path("next_question_set/", views.user_next_question_set, name="user_next_question_set"),
    path("review/<int:question_set_id>/", views.practice_review, name='practice_review'),
    path("review_data/<int:question_set_id>/", views.retrieve_review_data, name='practice_review_data'),
]