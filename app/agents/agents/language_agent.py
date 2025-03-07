import os
import openai

class LanguageAgent:
    """
    Agent responsible for analyzing the clarity, grammar, and structure of student answers.
    """
    def __init__(self, model_name: str = "NousResearch/DeepHermes-3-Llama-3-8B-Preview"):
        self.llm = LLM(model_name=model_name)
        self.output_parser = JsonOutputParser()
        self.prompt = PromptTemplate(
            template=LANGUAGE_ANALYSIS_PROMPT,
            input_variables=["question", "student_answer", "correct_answer"]
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt, output_parser=self.output_parser)
    
    def analyze(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the language usage in a student's answer.
        
        Args:
            state: The current state containing question, student_answer, and correct_answer
            
        Returns:
            Updated state with language feedback
        """
        result = self.chain.run({
            "question": state["question"],
            "student_answer": state["student_answer"],
            "correct_answer": state["correct_answer"]
        })
        
        # Update state with language feedback
        state["feedback"]["language"] = result
        
        return state