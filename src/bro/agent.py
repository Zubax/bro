import asyncio
import logging
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

_logger = logging.getLogger(__name__)


class EmailWorkflowState(TypedDict):
    """State for the email checking workflow."""

    unread_emails: list[dict[str, Any]]
    processed_emails: list[dict[str, Any]]
    report: str
    current_step: str
    error: str | None
    workflow_context: str  # Store the workflow prompt for reference


class EmailAgent:
    """LangGraph-based email agent with structured workflow to ensure all steps are followed."""

    def __init__(
        self,
        langchain_gmail: Any,
        langchain_shopify: Any,
        llm: Any,
        workflow_prompt: str,
    ):
        """Initialize the email agent.

        Args:
            langchain_gmail: LangChain MultiServerMCPClient for Gmail
            langchain_shopify: LangChain MultiServerMCPClient for Shopify
            llm: LangChain LLM instance
            workflow_prompt: Email workflow prompt text
        """
        self._llm = llm
        self._workflow_prompt = workflow_prompt
        self._langchain_gmail = langchain_gmail
        self._langchain_shopify = langchain_shopify

        # Fetch tools once during initialization
        gmail_tools = asyncio.run(langchain_gmail.get_tools())
        shopify_tools = asyncio.run(langchain_shopify.get_tools())
        self._tools = gmail_tools + shopify_tools

        self._workflow = self._build_workflow()
        _logger.info("EmailAgent initialized")

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow that ensures all steps are executed."""
        workflow = StateGraph(EmailWorkflowState)

        # Define the workflow steps
        workflow.add_node("search_unread", self._search_unread_emails)
        workflow.add_node("check_if_empty", self._check_if_empty)
        workflow.add_node("categorize_and_process", self._categorize_and_process_emails)
        workflow.add_node("generate_report", self._generate_report)

        # Define the flow
        workflow.set_entry_point("search_unread")
        workflow.add_edge("search_unread", "check_if_empty")
        workflow.add_conditional_edges(
            "check_if_empty",
            lambda state: "empty" if not state["unread_emails"] else "process",
            {"empty": "generate_report", "process": "categorize_and_process"},
        )
        workflow.add_edge("categorize_and_process", "generate_report")
        workflow.add_edge("generate_report", END)

        return workflow.compile()

    def _search_unread_emails(self, state: EmailWorkflowState) -> EmailWorkflowState:
        """Step 1: Search for unread emails - LLM decides which tools to use."""
        _logger.info("Step 1: Searching for unread emails...")
        state["current_step"] = "search_unread"

        try:
            messages = [
                SystemMessage(
                    content="You are an email processing assistant. Search for unread emails using Gmail search tools with query: is:unread"
                ),
                HumanMessage(content="Search for unread emails now."),
            ]
            response = self._llm.bind_tools(self._tools).invoke(messages)

            # Execute tool calls if LLM decided to use tools
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    tool = next((t for t in self._tools if t.name == tool_call["name"]), None)
                    if tool:
                        result = tool.invoke(tool_call["args"])
                        state["unread_emails"] = self._parse_search_results(result)
                        _logger.info(f"Found {len(state['unread_emails'])} unread emails")

        except Exception as e:
            _logger.error(f"Error searching emails: {e}")
            state["error"] = str(e)

        return state

    def _check_if_empty(self, state: EmailWorkflowState) -> EmailWorkflowState:
        """Step 2: Check if there are any unread emails."""
        _logger.info("Step 2: Checking if unread emails exist...")
        state["current_step"] = "check_if_empty"
        return state

    def _categorize_and_process_emails(self, state: EmailWorkflowState) -> EmailWorkflowState:
        """Step 3: Categorize and process each email - LLM decides actions."""
        _logger.info("Step 3: Categorizing and processing emails...")
        state["current_step"] = "categorize_and_process"

        processed = []
        for email in state["unread_emails"]:
            try:
                # Ask LLM to categorize and process the email using available tools
                # Include workflow context for decision making
                messages = [
                    SystemMessage(
                        content=f"You are an email processing assistant. Follow this workflow:\n\n{state['workflow_context']}\n\nCategorize and process the email below."
                    ),
                    HumanMessage(content=f"Email to process:\n{email}"),
                ]
                response = self._llm.bind_tools(self._tools).invoke(messages)

                # Execute tool calls if LLM decided to use tools
                actions = []
                if response.tool_calls:
                    for tool_call in response.tool_calls:
                        tool = next((t for t in self._tools if t.name == tool_call["name"]), None)
                        if tool:
                            result = tool.invoke(tool_call["args"])
                            actions.append(f"{tool_call['name']}: {result}")

                # Explicitly mark email as read after processing
                mark_read_tool = next((t for t in self._tools if "modify_gmail_message_labels" in t.name), None)
                if mark_read_tool and "id" in email:
                    try:
                        mark_read_tool.invoke({"message_id": email["id"], "remove_label_names": ["UNREAD"]})
                        actions.append(f"Marked as read: {email.get('subject', 'Unknown')}")
                    except Exception as e:
                        _logger.warning(f"Failed to mark email as read: {e}")

                processed.append(
                    {"email": email, "action": response.content or "\n".join(actions), "status": "processed"}
                )
            except Exception as e:
                _logger.error(f"Error processing email: {e}")
                processed.append({"email": email, "error": str(e), "status": "failed"})

        state["processed_emails"] = processed
        return state

    def _generate_report(self, state: EmailWorkflowState) -> EmailWorkflowState:
        """Step 4: Generate final report."""
        _logger.info("Step 4: Generating report...")
        state["current_step"] = "generate_report"

        report = "EMAIL CHECK RESULTS:\n"

        if state.get("error"):
            report += f"Error: {state['error']}\n"
        elif not state["unread_emails"]:
            report += "No new emails found.\n"
        else:
            report += f"Processed {len(state['processed_emails'])} emails:\n"
            for item in state["processed_emails"]:
                if item["status"] == "processed":
                    report += f"- {item['action']}\n"
                else:
                    report += f"- Failed: {item.get('error', 'Unknown error')}\n"

        state["report"] = report
        return state

    def _parse_search_results(self, result: Any) -> list[dict[str, Any]]:
        """Parse search results from Gmail tool."""
        # TODO: Implement proper parsing based on actual tool output format
        if isinstance(result, str):
            # Placeholder parsing logic
            return [{"id": "1", "subject": "Example", "from": "user@example.com"}]
        return []

    def check_emails(self) -> str:
        """Execute the email checking workflow.

        Returns:
            A formatted report of the email check results
        """
        _logger.info("Starting email check workflow with LangGraph...")

        try:
            # Initialize state with workflow context provided upfront
            initial_state: EmailWorkflowState = {
                "unread_emails": [],
                "processed_emails": [],
                "report": "",
                "current_step": "",
                "error": None,
                "workflow_context": self._workflow_prompt,  # Provide workflow prompt once
            }

            # Run the workflow
            final_state = self._workflow.invoke(initial_state)

            return final_state["report"]

        except Exception as e:
            _logger.error(f"Error in email check workflow: {e}", exc_info=True)
            return f"EMAIL CHECK RESULTS:\nError during email check: {e}"
