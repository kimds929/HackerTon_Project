from django.db import connection
import random
from gen_ai.gen_ai import problem_generation_pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed

#def mapping_generation_pipeline():
def mapping_generation_pipeline(target_chapter_id, target_difficulty, target_problem_num=5):
    """
    @param: target_chapter_id, target_difficulty(-1, 0, 1)
    @return: generated problems(list of dictionaries)
    """
    # suppose that we have a recommendation model
    # TODO change here!
    # target_chapter_id, target_difficulty = 468, 1

    find_keyconcept_query = """
        SELECT ck.kc_id
        FROM chapter_keyconcept_conj ck
        WHERE ck.chapter_id = %s
        """

    with connection.cursor() as cursor:
        cursor.execute(find_keyconcept_query, [target_chapter_id])
        target_keyconcept_ids = cursor.fetchall()

    # Find question topics from the key concepts
    keyconcept_questiontopic_dict = dict()
    for keyconcept_id in target_keyconcept_ids:
        find_questiontopic_query = """
            SELECT kq.question_topic_id
            FROM question_topic_kc_conj kq
            WHERE kq.kc_id = %s
            """
        
        with connection.cursor() as cursor:
            cursor.execute(find_questiontopic_query, [keyconcept_id[0]])
            questiontopic_ids = cursor.fetchall()
        
        keyconcept_questiontopic_dict[keyconcept_id[0]] = []
        for questiontopic_id in questiontopic_ids:
            keyconcept_questiontopic_dict[keyconcept_id[0]].append(questiontopic_id[0])
    
    # Check if all items in the dictionary are empty
    if all([len(v) == 0 for v in keyconcept_questiontopic_dict.values()]):
        # back to recommendation model
        return False
    
    # traverse the values of the dictionary one item per key
    question_list = []

    max_len = max(len(question_topic) for question_topic in keyconcept_questiontopic_dict.values())
    for i in range(max_len):
        for keyconcept, question_topic in keyconcept_questiontopic_dict.items():
            if i < len(question_topic):
                find_question_query = """
                    SELECT q.question_id, q.question_difficulty, q.question_text
                    FROM questions q
                    WHERE q.question_topic = %s
                    and q.question_type1 = '선택형'
                    and q.figure_text = ''
                    and q.generated = false
                    """
                
                with connection.cursor() as cursor:
                    cursor.execute(find_question_query, [question_topic[i]])
                    questions = cursor.fetchall()
                                
                if questions:
                    for question in questions:
                        question_list.append((keyconcept, question, question_topic[i]))
    # pick five questions randomly
    # if there are less than five questions,
    # do restorative sampling to get five questions
    if len(question_list) >= target_problem_num:
        question_list = random.sample(question_list, target_problem_num)
    else:
        question_list += random.sample(question_list, target_problem_num - len(question_list))

    # generate problems with the selected problems
    # this can be done in a multi-threaded way
    max_threads = 5
    generated_problems = []
    with ThreadPoolExecutor(max_threads) as executor:
        futures = {executor.submit(problem_generation_pipeline, question[1][2], question[1][1], target_difficulty): question for question in question_list}

        for future in as_completed(futures):
            question = futures[future]
            result = future.result()
            result["kc_id"] = question[0]
            result["question_topic"] = question[2]
            generated_problems.append(result)
    
    return generated_problems
    
            
def mapping_generation_pipeline_initial(target_chapter_ids, target_difficulty, target_problem_num=5):
    """
    @param: target_chapter_ids, target_difficulty(-1, 0, 1)
    @return: generated problems(list of dictionaries)
    """
    # suppose that we have a recommendation model
    # TODO change here!
    # target_chapter_id, target_difficulty = 468, 1

    find_keyconcept_query = """
        SELECT ck.kc_id
        FROM chapter_keyconcept_conj ck
        WHERE ck.chapter_id IN %s
        """

    target_chapter_ids_tuple = tuple(target_chapter_ids)

    with connection.cursor() as cursor:
        cursor.execute(find_keyconcept_query, [target_chapter_ids_tuple])
        target_keyconcept_ids = cursor.fetchall()

    # Find question topics from the key concepts
    keyconcept_questiontopic_dict = dict()
    for keyconcept_id in target_keyconcept_ids:
        find_questiontopic_query = """
            SELECT kq.question_topic_id
            FROM question_topic_kc_conj kq
            WHERE kq.kc_id = %s
            """
        
        with connection.cursor() as cursor:
            cursor.execute(find_questiontopic_query, [keyconcept_id[0]])
            questiontopic_ids = cursor.fetchall()
        
        keyconcept_questiontopic_dict[keyconcept_id[0]] = []
        for questiontopic_id in questiontopic_ids:
            keyconcept_questiontopic_dict[keyconcept_id[0]].append(questiontopic_id[0])
    
    # Check if all items in the dictionary are empty
    if all([len(v) == 0 for v in keyconcept_questiontopic_dict.values()]):
        # back to recommendation model
        return False
    
    # traverse the values of the dictionary one item per key
    question_list = []

    max_len = max(len(question_topic) for question_topic in keyconcept_questiontopic_dict.values())
    for i in range(max_len):
        for keyconcept, question_topic in keyconcept_questiontopic_dict.items():
            if i < len(question_topic):
                find_question_query = """
                    SELECT q.question_id, q.question_difficulty, q.question_text
                    FROM questions q
                    WHERE q.question_topic = %s
                    and q.question_type1 = '선택형'
                    and q.figure_text = ''
                    and q.generated = false
                    """
                
                with connection.cursor() as cursor:
                    cursor.execute(find_question_query, [question_topic[i]])
                    questions = cursor.fetchall()
                                
                if questions:
                    for question in questions:
                        question_list.append((keyconcept, question, question_topic[i]))
    # pick ten questions randomly
    # if there are less than five questions,
    # do restorative sampling to get five questions
    if len(question_list) >= target_problem_num:
        question_list = random.sample(question_list, target_problem_num)
    else:
        question_list += random.sample(question_list, target_problem_num - len(question_list))

    # generate problems with the selected problems
    # this can be done in a multi-threaded way
    max_threads = 5
    generated_problems = []
    with ThreadPoolExecutor(max_threads) as executor:
        futures = {executor.submit(problem_generation_pipeline, question[1][2], question[1][1], target_difficulty): question for question in question_list}

        for future in as_completed(futures):
            question = futures[future]
            result = future.result()
            result["kc_id"] = question[0]
            result["question_topic"] = question[2]
            generated_problems.append(result)
    
    return generated_problems
    
            
                