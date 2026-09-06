\# Personal Banking AI Agent



A hands-on project for building a secure, production-oriented AI banking support agent from the ground up.



The project starts with a simple LLM-powered banking assistant and progressively evolves toward a production-grade agentic architecture covering:



\- AI agents and tool calling

\- Banking system design

\- Authentication and authorization

\- Session and conversation memory

\- Multi-agent architecture

\- Model Context Protocol (MCP)

\- PII protection

\- Prompt-injection defense

\- Guardrails

\- Agent evaluation

\- Observability and tracing

\- Cost and runaway-loop protection

\- Human-in-the-loop workflows

\- Asynchronous processing

\- Docker and cloud deployment



The goal is not simply to build a chatbot.



The goal is to understand \*\*how to design, secure, evaluate, and deploy AI agent systems\*\*.



\---



\## 🚧 Project Status



This project is being built incrementally.



\### Current milestone



The current implementation includes:



\- FastAPI backend

\- Google Gemini integration

\- `/chat` API

\- LLM-powered conversations

\- Function/tool calling

\- Account balance retrieval

\- Transaction retrieval

\- Identity-bound banking tools

\- Basic authorization boundary

\- In-memory conversation sessions

\- Transaction analysis through the AI agent



The system is currently a \*\*learning/development implementation\*\* and is \*\*not a production banking system\*\*.



No real customer or banking data should be used.



\---



\# 🎯 Problem



Banking support receives a large number of repetitive customer requests.



Typical requests include:



1\. "What's my balance?"

2\. "What was this debit from my account?"

3\. "Send me a cheque book."



A banking AI assistant can potentially handle many of these requests automatically.



However, banking introduces requirements that ordinary chatbots do not have:



\- Strong identity verification

\- Authorization

\- Customer-data isolation

\- PII protection

\- Secure tool execution

\- Transaction safety

\- Auditability

\- Prompt-injection resistance

\- Reliable evaluation

\- Human approval for risky actions

\- Production observability



This project explores how to build such a system step by step.



\---



\# 🏗️ Current Architecture



The current system is intentionally simple:



```text

\&#x20;                        Customer

\&#x20;                           |

\&#x20;                           v

\&#x20;                      Chat Request

\&#x20;                           |

\&#x20;                           v

\&#x20;                      FastAPI API

\&#x20;                           |

\&#x20;                           v

\&#x20;                      Gemini LLM

\&#x20;                           |

\&#x20;               +-----------+-----------+

\&#x20;               |                       |

\&#x20;               v                       v

\&#x20;       Balance Tool           Transactions Tool

\&#x20;               |                       |

\&#x20;               v                       v

\&#x20;       Banking Data             Banking Data

\&#x20;               |

\&#x20;               v

\&#x20;       Authenticated Client


