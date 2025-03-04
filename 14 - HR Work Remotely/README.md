# HR Work Remotely: Multi-Agent LLM Workflow Handling Work Remotely Requests By Employees

In this example we demonstrate, how to use a multi-agent defense framework that allows the employees of a company to request to work remotely for few days.

# Remote Work Request System Flow

## Overview
This document outlines the flow of a remote work request system implemented through swarm agents in AG2. The system allows employees to request to work remotely, verifies their eligibility based on department policies, and optionally notifies their managers.

## Agents

### HR Triage Agent
- **Role**: Central coordinator of the workflow
- **Functions**: 
  - Records number of remote days (`remote_days`)
- **System Message Updates**:
  - Tracks customer name, login status, approved days, and manager notification preference
- **Decision Logic**:
  - Routes to Authentication Agent when user is not logged in
  - Routes to Remote Policy Agent when user is logged in
  - Routes to Email Agent for manager notification handling
  - Reverts to user after completing processes

### Authentication Agent
- **Role**: Verifies employee identity
- **Functions**:
  - Authenticates users via username (`login_by_username`)
- **Key Actions**:
  - Retrieves user data from database (name, department)
  - Records department manager information
  - Sets logged_in flag in context
- **Decision Logic**:
  - Routes back to HR Triage Agent once authentication is complete
  - Reverts to user if unable to authenticate

### Remote Policy Agent
- **Role**: Checks if remote work request complies with department policies
- **Functions**:
  - Records number of remote days (`remote_days`)
  - Validates remote day requests against policy (`check_remote_policy`)
- **System Message Updates**:
  - Tracks user login status and requested number of remote days
- **Decision Logic**:
  - Routes back to HR Triage Agent after policy verification
  - Reverts to user if necessary

### Email Agent
- **Role**: Handles manager notification
- **Functions**:
  - Checks if user wants to notify manager (`wants_to_notify`)
  - Sends notification email via Gmail API (`gmail_send_function`)
- **System Message Updates**:
  - Guides process for manager notification
- **Decision Logic**:
  - Routes back to HR Triage Agent after email handling

## Workflow Stages and Handoffs

1. **Initial Contact**
   - User contacts system about remote work
   - HR Triage Agent engages and evaluates login status

2. **Authentication**
   - If `requires_login` is True → HR Triage Agent → Authentication Agent
   - Authentication Agent verifies username against USER_DATABASE
   - Once `logged_in` is True → Authentication Agent → HR Triage Agent

3. **Remote Work Policy Check**
   - If user is authenticated (`logged_in` is True) → HR Triage Agent → Remote Policy Agent
   - Remote Policy Agent records requested days and checks against department policy
   - Remote Policy Agent validates days against DEPARTMENT_DATABASE
   - Once days are approved (`days_approved` is True) → Remote Policy Agent → HR Triage Agent

4. **Manager Notification**
   - If days are approved → HR Triage Agent → Email Agent
   - Email Agent asks if manager should be notified
   - If `notify_manager` is True, sends email to manager
   - Once email is sent (`email_sent` is True) → Email Agent → HR Triage Agent

5. **Completion**
   - HR Triage Agent confirms process completion
   - System returns to user interaction

## Context Variables

The workflow tracks the following contextual information:
- `user_name`: Full name of the employee
- `logged_in_username`: Username of the authenticated user
- `logged_in`: Authentication status flag
- `requires_login`: Flag indicating authentication requirement
- `remote_days`: Number of requested remote work days
- `has_defined_days`: Flag indicating if days have been specified
- `department`: Employee's department
- `defined_department`: Flag indicating if department is known
- `manager_name`: Name of the department manager
- `manager_email`: Email of the department manager
- `days_approved`: Flag indicating remote day request approval
- `notify_manager`: Flag indicating manager notification preference
- `email_sent`: Flag indicating successful email delivery

## Databases

The system utilizes two primary databases:

### User Database
Contains employee information:
- Username
- Full name
- Department

### Department Database
Contains department policies:
- Manager name and email
- Maximum allowed remote days
- Department address

## Flow Diagram

```
Customer → HR Triage Agent
    ↓
    ├── If not logged in → Authentication Agent → Back to HR Triage
    ↓
    ├── If logged in → Remote Policy Agent → Back to HR Triage
    ↓
    └── If days approved → Email Agent → Back to HR Triage → Customer
```