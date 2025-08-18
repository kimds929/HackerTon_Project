
// This script is used to fetch the problem set for the logged-in user
document.addEventListener("DOMContentLoaded", () => {
    console.log(nextProblemSetReady)
    if (!nextProblemSetReady) {
        alert("다음 문제 세트가 아직 준비되지 않았습니다. 홈으로 이동합니다.");
        window.location.href = '/';  // Redirect to home page
        return;  // Stop further execution
    }

    const userId = document.getElementById('userId') ? document.getElementById('userId').value : null;

    if (userId) {
        fetch('/practice/retrieve_set/', { // Ensure the URL is correct and matches your Django path
            method: 'POST', // Change the method to POST
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value, // Include CSRF token for Django
            },
            body: JSON.stringify({ user_id: userId }) // Send the user ID in the request body
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json(); // Parse response as JSON
        })
        .then(data => {
            if (data.success && data.questions) {
                const problemContainer = document.getElementById('problemContainer');
                problemContainer.innerHTML = ''; // Clear container

                data.questions.forEach((problem, index) => {
                    const problemItem = document.createElement('div');
                    problemItem.classList.add('problem-item');
                    problemItem.innerHTML = `
                        <h3>문제 ${index + 1}</h3>
                        <p>${problem.question_text}</p>
                        <input type="text" placeholder="답안을 입력하세요" data-problem-id="${problem.question_id}">
                    `;
                    problemContainer.appendChild(problemItem);
                    MathJax.typesetPromise()
                });
            } else {
                alert(data.message || "문제를 불러오지 못했습니다. 다시 시도해 주세요.");
            }
        })
        .catch(error => {
            console.error('Error fetching problem set:', error);
            alert("문제를 불러오는 도중 오류가 발생했습니다.");
        });
    } else {
        alert("사용자 ID를 확인할 수 없습니다.");
    }
});

// This script is used to submit the answers for the logged-in user
function submitAnswers() {
    const answers = [];
    document.querySelectorAll('.problem-item input').forEach(input => {
        answers.push({
            question_id: input.getAttribute('data-problem-id'),
            answer: input.value.trim()
        });
    });
    

    fetch('/practice/submit_answer/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value,
        },
        body: JSON.stringify({ answers })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert("답안이 제출되었습니다!");
            const questionSetId = data.question_set_id;
            window.location.href = `/practice/review/${questionSetId}`; // Redirect or refresh page
        } else {
            alert("답안 제출에 실패했습니다. 다시 시도해 주세요.");
        }
    })
    .catch(error => {
        console.error('Error submitting answers:', error);
        alert("답안을 제출하는 도중 오류가 발생했습니다.");
    });
}