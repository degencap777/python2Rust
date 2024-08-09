use py2rs::text_chunker;

// #[test]
// fn test_read_text_file() {
//     let document_uri = "books/book1.txt"; 
//     let expected_output = "A text chunker breaks large bodies of text into smaller. that are manageable pieces."; 

//     match text_chunker::read_text_file(document_uri, true) {
//         Ok(content) => assert_eq!(content, expected_output),
//         Err(e) => panic!("Failed to read text file: {}", e),
//     }
// }

// #[test]
// fn test_split_lines() {
//     let original_text = "First line.\nSecond line.\rThird line.";
//     let goal = ["First line.", "Second line.", "Third line."];
//     let result: Vec<&str> = text_chunker::split_lines(original_text);
//     assert_eq!(result, goal)
// }

// #[test]
// fn test_fix_line_breaks() {
//     let original_text = "First line.\nSecond line.\rThird line.";
//     let goal = "First line.\nSecond line.\nThird line.";
//     let result: String = text_chunker::fix_line_breaks(original_text);
//     assert_eq!(result, goal)
// }

// #[test]
// fn fix_apostrophes() {
    
// }

// #[test]
// fn test_normalize_special_characters() {

// }