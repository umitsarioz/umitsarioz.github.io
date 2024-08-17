---
# the default layout is 'page'
icon: fas fa-info-circle
order: 4
---


<section style="
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 20px;
">
    <div style="
        display: flex;
        flex-direction: column;
        align-items: center;
        border-radius: 10px;
        padding: 20px;
        max-width: 800px;
        width: 100%;
        box-sizing: border-box;
    ">
        <img src="/assets/img/core/avatar.jpeg" alt="Profil Fotoğrafı" style="
            border-radius: 50%;
            width: 250px;
            height: 250px;
            object-fit: cover;
            margin-bottom: 20px;
        ">
        <div style="
            font-size: 1rem;
            line-height: 1.6;
            text-align: justify;
            padding: 0 20px; /* Ensures padding on both sides */
            box-sizing: border-box; /* Prevents padding from affecting the width */
            font-style: italic; /* Italicizes the text */
        "> <p>Ümit has a BSc degree in Computer Engineering from Gazi University. He has more than 3 years of professional experience in data related jobs such as Artificial Intelligence Engineer, Software Engineer and Data Scientist. He is good at Python and has a good understanding of supervised and unsupervised machine learning algorithms. He has experience especially in problems such as time series forecasting, clustering, graph structures, anomaly detection and classification. He also has experience with big data tools such as Apache Spark, Kafka, Hadoop (HDFS), Pandas, Numpy, NetworkX, Scikit Learn, Plotly, Matplotlib, Airflow and databases such as Mongo, Cassandra, Postgresql. His interests are big data processing and machine learning algorithms. He is currently doing research on clustering methods, evaluations and LLMs.</p>
        </div>
    </div>
</section>

<style>
/* Responsive Design */
@media (min-width: 768px) {
    section > div {
        flex-direction: row;
        align-items: flex-start;
    }

    section > div > div {
        text-align: left;
    }
}
</style>


