import uvicorn

from src.dependency_injector import DependencyInjector
from src.rag_service.presentation.api import create_app


def main() -> None:
    dependency_injector = DependencyInjector()

    ingest_use_case = dependency_injector.create_ingest_pdf_use_case()
    ask_question_use_case = dependency_injector.create_ask_question_use_case()

    app = create_app(ingest_use_case, ask_question_use_case)

    # Start server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )


if __name__ == "__main__":
    main()

