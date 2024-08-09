use py2rs::text_chunker;

#[test]
fn test_split_lines() {
    let original_text = "First line.\nSecond line.\rThird line.";
    let goal = ["First line.", "Second line.", "Third line."];
    let result: Vec<&str> = text_chunker::split_lines(original_text);
    assert_eq!(result, goal)
}

// #[test]
// fn test_fix_line_breaks() {
//     let original_text = "First line.\nSecond line.\rThird line.";
//     let goal = "First line.\nSecond line.\nThird line.";
//     let result: String = text_chunker::fix_line_breaks(original_text);
//     assert_eq!(result, goal)
// }

#[test]
fn fix_apostrophes() {
    let goal = "Here's an example text with fancy apostrophes: It's time to test this. Don't worry about it! She said, It's great!";
    let temp = goal.replace("'", "\\u2019");
    let result = text_chunker::fix_apostrophes(&temp);
    assert_eq!(result, goal)
}

// #[test]
// fn test_normalize_special_characters() {
//     let goal = "First 'line'.\nSecond 'line'.\nThird 'line'.";
//     let temp = "First \\u2019line\\u2019.\nSecond \\u2019line\\u2019.\rThird \\u2019line\\u2019.";
//     let result = text_chunker::normalize_special_characters(temp);
//     assert_eq!(result, goal)
// }

#[test]
fn test_read_text_file() {
    let path = "books/book1.txt"; 
    let goal = "First 'line'.\nSecond 'line'.\nThird 'line'."; 

    match text_chunker::read_text_file(path, true) {
        Ok(content) => assert_eq!(content, goal),
        Err(e) => panic!("Failed to read text file: {}", e),
    }
}

#[test]
fn test_split_into_sentences() {
    let goal = ["First 'line'.", "Second 'line'.", "Third 'line'."];
    let temp = "First 'line'.\nSecond 'line'.\nThird 'line'.";
    let result = text_chunker::split_into_sentences(temp);
    assert_eq!(result, goal)
}

#[test]
fn test_create_nonoverlapping_chunks() {
    let goal = ["First 'line'.", "Second 'line'.", "Third 'line'."];
    let temp = "First 'line'.\nSecond 'line'.\nThird 'line'.";
    let result = text_chunker::create_nonoverlapping_chunks(temp, 10);
    assert_eq!(result, goal)
}

#[test]
fn test_merge_chunks() {
    let goal = ["First 'line'.\nSecond 'line'.", "Second 'line'.\nThird 'line'.", "Third 'line'."];
    let temp:Vec<String> = vec!["First 'line'.".to_string(), "Second 'line'.".to_string(), "Third 'line'.".to_string()];
    let mut chunk_strings: Vec<String> = Vec::new();
    let overlap_factor = 2;
    for i in (0..temp.len()).step_by(overlap_factor - 1) {
        chunk_strings.push(text_chunker::_merge_chunks(&temp, i, overlap_factor));
    }
    assert_eq!(chunk_strings, goal)
}

#[test]
fn test_get_text_chunks() {
    let goal = "Bundle[2[13]: First 'line'.], Bundle[2[14]: Second 'line'.], Bundle[2[13]: Third 'line'.]";
    let temp = "First 'line'.\nSecond 'line'.\nThird 'line'.";
    let result = text_chunker::get_text_chunks("book1.txt", temp, 5, &mut 0, true);
    let result_string = result.into_iter()
                        .map(|item| item.to_string())
                        .collect::<Vec<String>>()  // Collect into a Vec<String>
                        .join(", ");
    assert_eq!(result_string, goal)
}
