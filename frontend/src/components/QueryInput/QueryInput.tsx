import React, { useState } from 'react';

interface QueryInputProps {
  onGenerate: (question: string) => void;
  isLoading: boolean;
  disabled?: boolean;
}

const EXAMPLE_QUERIES = [
  'Show all students with CGPA above 8.5',
  'Find courses with credits equal to 4',
  'List students whose department is CSE',
  'Show all courses ordered by credits descending',
  'Find enrollments where marks are greater than 85',
  'Get students whose age is 20 and CGPA is above 8',
];

export const QueryInput: React.FC<QueryInputProps> = ({
  onGenerate,
  isLoading,
  disabled = false,
}) => {
  const [question, setQuestion] = useState('');
  const [validationError, setValidationError] = useState<string | null>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = question.trim();
    if (!trimmed) {
      setValidationError('Please enter a natural language question.');
      return;
    }
    setValidationError(null);
    onGenerate(trimmed);
  };

  const handleSelectExample = (example: string) => {
    setQuestion(example);
    setValidationError(null);
    if (!isLoading && !disabled) {
      onGenerate(example);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setQuestion(e.target.value);
    if (validationError) {
      setValidationError(null);
    }
  };

  return (
    <section className="query-card" data-testid="query-input-section">
      <div className="query-card-header">
        <div className="section-title">
          <span>💬</span>
          <span>Natural Language Query</span>
        </div>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="input-container">
          <input
            type="text"
            className="query-input"
            placeholder="Ask a question about students, courses, or enrollments in plain English..."
            value={question}
            onChange={handleChange}
            disabled={isLoading || disabled}
            aria-label="Natural language question"
            data-testid="question-input"
          />
          <button
            type="submit"
            className="primary-button"
            disabled={isLoading || disabled || !question.trim()}
            data-testid="generate-button"
          >
            {isLoading && <span className="spinner" />}
            <span>{isLoading ? 'Translating...' : 'Generate SQL'}</span>
          </button>
        </div>
      </form>

      {validationError && (
        <div className="banner banner-error" data-testid="input-validation-error">
          <span>⚠️</span>
          <span>{validationError}</span>
        </div>
      )}

      {/* Suggested Query Pills */}
      <div className="example-queries">
        <span className="example-label">Try an example question:</span>
        <div className="pills-row">
          {EXAMPLE_QUERIES.map((example) => (
            <button
              key={example}
              type="button"
              className="query-pill"
              onClick={() => handleSelectExample(example)}
              disabled={isLoading || disabled}
            >
              {example}
            </button>
          ))}
        </div>
      </div>
    </section>
  );
};
