# Implementation Summary

The task was to modify the chat interface so that each source becomes a clickable link that opens the corresponding lesson video in a new tab.

## Changes Made

1. Modified the `_format_results` method in `backend/search_tools.py` to include lesson links in the sources information.

2. The implementation:
   - Retrieves the lesson link from the vector store using the existing `get_lesson_link` method
   - Embeds the link invisibly in the source string using a `||` separator
   - Maintains backward compatibility with existing source formatting

## Result

The sources returned by the search now include lesson links embedded invisibly with the format:
`Course Title - Lesson Number||https://example.com/lesson-link`

The frontend can parse these sources to extract the lesson links and create clickable links that open in a new tab.

## Testing

Verified that the implementation works correctly by testing the query endpoint, which now returns sources with embedded lesson links.