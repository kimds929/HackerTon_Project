from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.http import require_POST, require_GET
from .models import *
from apps.p4ds.models import *
import json
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import BaseUserCreationForm
from django import forms
from django.views.decorators.csrf import csrf_exempt
import logging
from django.db import connection
from apps.p4ds.task import core_rec_gen_pipeline
from celery import shared_task

logger = logging.getLogger(__name__)

def practice_main(request):
    if request.user.is_authenticated:
        user = Users.objects.get(user_id=request.user.user_id)
        
        howmany_problems_query = """
            SELECT 
                COUNT(*) AS total_problems,
                SUM(CASE WHEN qh.is_correct = TRUE THEN 1 ELSE 0 END) AS correct_problems
            FROM question_history qh
            WHERE qh.user_no = %s
        """

        with connection.cursor() as cursor:
            cursor.execute(howmany_problems_query, [user.user_no])
            result = cursor.fetchone()
            howmany_problems = result[0]
            correct_problems = result[1]
        
        correct_rate = round(correct_problems / howmany_problems * 100, 2) if howmany_problems > 0 else 0        
        context = {
            'user_id' : user.user_id,
            'total_solved': howmany_problems,
            'success_rate': correct_rate
        }
        return render(request, 'practice/practice_main.html', context)
    else:
        return render(request, 'p4ds/login.html')

        
        
def practice_attempt(request):
    if request.user.is_authenticated:
        next_problem_set_ready = Users.objects.get(user_id=request.user.user_id).next_question_set_ready
        context = {
            'user_id': request.user.user_id,
            'next_problem_set_ready': next_problem_set_ready
        }
        return render(request, 'practice/practice_attempt.html', context)
    else:
        return render(request, 'p4ds/login.html')

@require_POST
def retrieve_practice_set(request):
    """API endpoint to get the next question set for the user"""
    try:
            
        body = request.body.decode('utf-8')
        logger.info("Request body: %s", body)     

        # 1. get input from JSON body
        data = json.loads(request.body)
        
        # 2. check required fields
        required_fields = ["user_id"]
        for field in required_fields:
            if field not in data:
                return JsonResponse({
                    "success": False,
                    "message": f"{field} 필수 필드가 누락되었습니다."
                }, status=400)
        
        # 3. retrieve data fields
        user_id = data.get("user_id")

        # 4. Check if user exists
        if Users.objects.filter(user_id=user_id).exists():
            user = Users.objects.get(user_id=user_id)
            # 5. Get the next question set
            next_question_set_id = user.next_question_set_id
            if next_question_set_id:
                
                query = """
                    SELECT gq.question_id, gq.question_difficulty, gq.question_text
                    FROM question_set_questions_conj qc
                    INNER JOIN generated_questions gq ON qc.question_id = gq.question_id
                    WHERE qc.question_set_id = %s
                """

                # 쿼리 실행
                with connection.cursor() as cursor:
                    cursor.execute(query, [next_question_set_id])
                    questions_data = cursor.fetchall()

                # Prepare data to return
                question_data = []
                for question in questions_data:
                    question_data.append({
                        "question_id": question[0],
                        "question_difficulty": question[1],
                        "question_text":question[2],
                    })
                
                # Return the question data as part of the response
                return JsonResponse({
                    "success": True,
                    "questions": question_data
                }, status=200)
        else:
            # If user_id does not exist, return error response
            return JsonResponse({
                "success": False,
                "message": "존재하지 않는 사용자입니다."
            }, status=400)
    except Exception as e:
        # General error response
        return JsonResponse({
            "success": False,
            "message": str(e)
        }, status=400)    

#@require_POST
def submit_answer(request):
    try:
            
        body = request.body.decode('utf-8')
        logger.info("Request body: %s", body)     

        # 1. get input from JSON body
        data = json.loads(request.body)
        user_id = request.user.user_id
        
        # 2. check required fields
        required_fields = ["answers"]
        for field in required_fields:
            if field not in data:
                return JsonResponse({
                    "success": False,
                    "message": f"{field} 필수 필드가 누락되었습니다."
                }, status=400)
        
        # 3. retrieve data fields
        question_answer_pair_list = data.get("answers")
        # 사용자 확인
        user = Users.objects.filter(user_id=user_id).first()
        if not user:
            return JsonResponse({
                "success": False,
                "message": "존재하지 않는 사용자입니다."
            }, status=400)

        user = Users.objects.get(user_id=user_id)
        problem_set_id = user.next_question_set_id # 문제 푸는 사이에 건드리면 안된다.
            
        # 문제집 존재 여부 확인 - 불필요?
        # question_set = QuestionSet.objects.filter(question_set_id=problem_set_id).first()
        # if not question_set:
        #     return JsonResponse({
        #         "success": False,
        #         "message": "존재하지 않는 문제집입니다."
        #     }, status=400)
        
        # question + answer select
        question_ids = [qa_pair["question_id"] for qa_pair in question_answer_pair_list]
        formatted_question_ids = ','.join([f"'{qid}'" for qid in question_ids])
        query = f"""
                SELECT gq.question_id, ga.answer_correct
                FROM generated_answers ga
                INNER JOIN generated_questions gq ON ga.question_id = gq.question_id
                WHERE ga.question_id IN ({formatted_question_ids})
            """

        with connection.cursor() as cursor:
            cursor.execute(query)
            answers_data = cursor.fetchall()

        print(answers_data)
        print(question_answer_pair_list)
        # 채점
        response_data = []
        for qa_pair in question_answer_pair_list:
            question_id = qa_pair["question_id"]
            user_answer = int(qa_pair['answer'])

            matching_answer = next((answer for answer in answers_data if answer[0] == int(question_id)), None)

            answer_correct = int(matching_answer[1])
            user_correct = (user_answer == answer_correct)
    
            # 필요한가?
            response_data.append({
                "question_id": matching_answer[0],
                "user_correct" : user_correct
            })

            # question_history insert
            query = f"""
                    INSERT INTO public.question_history
                    (question_set_id, question_id, user_no, is_correct, user_answer)
                    VALUES({problem_set_id}, '{question_id}', {user.user_no}, {user_correct}, '{user_answer}');
                     """    
            
            with connection.cursor() as cursor:
                cursor.execute(query)

        print(user.user_id)
        
        # 현재의 문제집을 종료시키는 처리
        # users next_quesiton_set_id null, next_question_set_ready false update
        query = f"""
                    UPDATE public.users
                    SET next_question_set_id=null, next_question_set_ready=false 
                    WHERE user_no = {user.user_no};
                    """    
        
        
        with connection.cursor() as cursor:
            cursor.execute(query)

        # users_question_set_conj question_set_status 3 update
        query = f"""
                    UPDATE public.users_question_set_conj
                    SET question_set_status=3
                    WHERE user_no = {user.user_no} and question_set_id = {problem_set_id};
                    """    
        
        with connection.cursor() as cursor:
            cursor.execute(query)
        
        # trigger recommendation + problem generation 
        # 추천을 위해 넘겨주어야 할 것 
        core_rec_gen_pipeline.delay(user.user_no)

        return JsonResponse({
            "success": True,
            "data": response_data,
            "question_set_id": problem_set_id
        }, status=200)
    
    except Exception as e:
        # General error response
        print(e)
        return JsonResponse({
            "success": False,
            "message": str(e)
        }, status=400)    

def practice_review(request, question_set_id):
    if request.user.is_authenticated:
        return render(request, 'practice/practice_review.html',
                      {'question_set_id': question_set_id})
    else:
        return render(request, 'p4ds/login.html')

def retrieve_review_data(request, question_set_id):
    try:
        review_query = """
            SELECT
                gq.question_text,
                ga.answer_correct,
                ga.answer_text,
                gh.user_answer,
                gh.is_correct
            FROM question_history gh
            INNER JOIN generated_questions gq ON gh.question_id = gq.question_id
            INNER JOIN generated_answers ga ON gh.question_id = ga.question_id
            WHERE gh.question_set_id = %s and gh.user_no = %s
        """
        with connection.cursor() as cursor:
            cursor.execute(review_query, [question_set_id, request.user.user_no])
            review_data = cursor.fetchall()
        
        response_data = []
        for row in review_data:
            response_data.append({
                "question_text": row[0],
                "answer_correct": row[1],
                "answer_text": row[2],
                "user_answer": row[3],
                "is_correct": row[4]
            })
        return JsonResponse({
            "success": True,
            "response_data": response_data
        }, status=200)
        
    except Exception as e:
        return JsonResponse({
            "success": False,
            "message": str(e)
        }, status=400)


@require_POST
def user_next_question_set(request):
    try:

            body = request.body.decode('utf-8')
            logger.info("Request body: %s", body)     

            # 1. get input from JSON body
            data = json.loads(request.body)
            
            # 2. check required fields
            required_fields = ["user_id"]
            for field in required_fields:
                if field not in data:
                    return JsonResponse({
                        "success": False,
                        "message": f"{field} 필수 필드가 누락되었습니다."
                    }, status=400)
            
            # 3. retrieve data fields
            user_id = data.get("user_id")

            # 4. Check if user exists
            if Users.objects.filter(user_id=user_id).exists():
                user = Users.objects.get(user_id=user_id)
                
                # 5. Prepare user data for response
                user_data = {
                    "success": True,
                    "next_question_set_id": user.next_question_set_id,
                    "next_question_set_ready": user.next_question_set_ready
                }
                return JsonResponse(user_data, status=200)

            else:
                # If user_id does not exist, return error response
                return JsonResponse({
                    "success": False,
                    "message": "존재하지 않는 사용자입니다."
                }, status=400)
    except Exception as e:
        # General error response
        return JsonResponse({
            "success": False,
            "message": str(e)
        }, status=400)           