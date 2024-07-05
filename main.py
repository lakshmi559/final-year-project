import streamlit as st
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class JobSearchAgent:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.job_files = self.load_job_files()
        self.vectorize_jobs()

    def load_job_files(self):
        job_files = []
        for file_name in os.listdir(self.folder_path):
            if file_name.endswith('.txt'):
                with open(os.path.join(self.folder_path, file_name), 'r') as file:
                    job_files.append(file.read())
        return job_files

    def vectorize_jobs(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.job_vectors = self.vectorizer.fit_transform(self.job_files)

    def search_jobs(self, query, top_n=3):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.job_vectors)
        top_indices = similarities.argsort()[0][-top_n:][::-1]
        return [self.job_files[index] for index in top_indices]

def main():
    st.title("Intelligent Job Search")

    folder_path = "jobs"  # Update the folder path as per your directory structure
    agent = JobSearchAgent(folder_path)

    query = st.text_area("Enter job details:")
    if st.button("Search"):
        if query:
            with st.spinner("Searching for matching jobs..."):
                results = agent.search_jobs(query)
                st.subheader("Top Matching Jobs:")
                for i, result in enumerate(results, 1):
                    st.markdown(f"**Job {i}:**")
                    st.write(result)
                    st.markdown("---")

if __name__ == "__main__":
    main()
