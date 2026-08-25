from airflow.sdk import DAG

from _balearic_mhws_dag_factory import build_dag

balearic_mhws_rep: DAG = build_dag("rep")
