#from config.celery import app
from django.db import connection
import time
import logging
from celery import shared_task
from recommender.get_rec import mapping_generation_pipeline, mapping_generation_pipeline_initial
from MathProject.Run_Inference import inference
from MathProject.Generate_InitialTest import Generate_InitialTestSet
from apps.p4ds.models import Users

logger = logging.getLogger(__name__)

@shared_task
def core_rec_gen_pipeline(user_no):
    print("Extracting User History")
    # we choose the recent 30 history
    extract_history_query = """
        SELECT qh.question_id, gq.kc_id, qh.is_correct, gq.question_difficulty
        from question_history qh
        JOIN generated_questions gq ON qh.question_id = gq.question_id
        where qh.user_no = %s
        ORDER BY qh.question_history_id DESC
        LIMIT 30
    """
    
    with connection.cursor() as cursor:
        cursor.execute(extract_history_query, [user_no])
        all_history = cursor.fetchall()
        
    user_history = []
    kc_ids = []
    is_corrects = []
    difficulties = []
    for history in all_history:
        kc_id = history[1]
        is_correct = history[2]
        if is_correct == True:
            is_correct = 1
        else:
            is_correct = 0
        difficulty = history[3] - 2
        kc_ids.append(kc_id)
        is_corrects.append(is_correct)
        difficulties.append(difficulty)
    user_history.append(kc_ids)
    user_history.append(is_corrects)
    user_history.append(difficulties)
    print("num of user history: ", len(user_history[0]))
    target_chapter_id, target_difficulty = inference(user_history)
    
    print('Reinforcement Learning Inference done')
    target_difficulty += 2

    # 추천 문제 생성
    generated_problems = mapping_generation_pipeline(target_chapter_id, target_difficulty)
    #generated_problems = mapping_generation_pipeline()
    # list of dictionaries
    # keys: problem_text, solution_text, answer, kc_id, question_topic
    print('Problem Generation done')
    
    question_ids = []
    # 생성된 문제 DB에 저장
    for problem in generated_problems:
        insert_problem_query = """
            INSERT INTO public.generated_questions
            (question_text, question_difficulty, kc_id)
            values(%s, %s, %s)
            RETURNING question_id
        """
        with connection.cursor() as cursor:
            cursor.execute(insert_problem_query, [problem['problem_text'], 2, problem['kc_id']])
            question_id = cursor.fetchone()[0]
            question_ids.append(question_id)
        
        insert_answer_query = """
            INSERT INTO public.generated_answers
            (question_id, answer_text, answer_correct)
            values(%s, %s, %s)
            """

        with connection.cursor() as cursor:
            cursor.execute(insert_answer_query, [question_id, problem['solution_text'], problem['answer']])

    # insert into problem set
    create_question_set_query = """
        INSERT INTO public.question_set
        DEFAULT VALUES
        RETURNING question_set_id
        """
    with connection.cursor() as cursor:
        cursor.execute(create_question_set_query)
        question_set_id = cursor.fetchone()[0]
    
    # insert into question_set_questions_conf(now the question_id is integer)
    insert_qs_questions_query = """
        INSERT INTO public.question_set_questions_conj
        (question_set_id, question_id)
        VALUES(%s, %s)
        """
    with connection.cursor() as cursor:
        for question_id in question_ids:
            cursor.execute(insert_qs_questions_query, [question_set_id, question_id])    
    
    # insert into users_question_set_conj
    insert_qs_user_query = """
        INSERT INTO public.users_question_set_conj
        (user_no, question_set_id, question_set_status)
        VALUES(%s, %s, %s)
        """
    with connection.cursor() as cursor:
        cursor.execute(insert_qs_user_query, [user_no, question_set_id, 2])

    # set user next_question_set_id, next_question_set_ready
    update_user_next_qs_query = """
        UPDATE public.users
        SET next_question_set_id = %s, next_question_set_ready = %s
        WHERE user_no = %s
        """
    with connection.cursor() as cursor:
        cursor.execute(update_user_next_qs_query, [question_set_id, True, user_no])

    print("Finished updating the user's next problem set.")

@shared_task
def initial_rec_gen_pipeline(user_no):

    target_difficulty = 0
    git = Generate_InitialTestSet(random_state=None)
    
    user = Users.objects.get(user_no=user_no)
    user_grade = user.user_grade
    target_chapter_ids = git.generate(user_grade, size=5)
    
    print('No inference for Initial test')
    target_difficulty += 2
    # 추천 문제 생성
    generated_problems = mapping_generation_pipeline_initial(target_chapter_ids, target_difficulty, target_problem_num=5)
    #generated_problems = mapping_generation_pipeline()
    # list of dictionaries
    # keys: problem_text, solution_text, answer, kc_id, question_topic
    print('Problem Generation done')
    
    question_ids = []
    # 생성된 문제 DB에 저장
    for problem in generated_problems:
        insert_problem_query = """
            INSERT INTO public.generated_questions
            (question_text, question_difficulty, kc_id)
            values(%s, %s, %s)
            RETURNING question_id
        """
        with connection.cursor() as cursor:
            cursor.execute(insert_problem_query, [problem['problem_text'], 2, problem['kc_id']])
            question_id = cursor.fetchone()[0]
            question_ids.append(question_id)
        
        insert_answer_query = """
            INSERT INTO public.generated_answers
            (question_id, answer_text, answer_correct)
            values(%s, %s, %s)
            """

        with connection.cursor() as cursor:
            cursor.execute(insert_answer_query, [question_id, problem['solution_text'], problem['answer']])

    # insert into problem set
    create_question_set_query = """
        INSERT INTO public.question_set
        DEFAULT VALUES
        RETURNING question_set_id
        """
    with connection.cursor() as cursor:
        cursor.execute(create_question_set_query)
        question_set_id = cursor.fetchone()[0]
    
    # insert into question_set_questions_conf(now the question_id is integer)
    insert_qs_questions_query = """
        INSERT INTO public.question_set_questions_conj
        (question_set_id, question_id)
        VALUES(%s, %s)
        """
    with connection.cursor() as cursor:
        for question_id in question_ids:
            cursor.execute(insert_qs_questions_query, [question_set_id, question_id])    
    
    # insert into users_question_set_conj
    insert_qs_user_query = """
        INSERT INTO public.users_question_set_conj
        (user_no, question_set_id, question_set_status)
        VALUES(%s, %s, %s)
        """
    with connection.cursor() as cursor:
        cursor.execute(insert_qs_user_query, [user_no, question_set_id, 2])

    # set user next_question_set_id, next_question_set_ready
    update_user_next_qs_query = """
        UPDATE public.users
        SET next_question_set_id = %s, next_question_set_ready = %s
        WHERE user_no = %s
        """
    with connection.cursor() as cursor:
        cursor.execute(update_user_next_qs_query, [question_set_id, True, user_no])

    print("Finished creating the user's first problem set.")


