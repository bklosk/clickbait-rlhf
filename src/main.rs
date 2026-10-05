use std::io;

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::layout::{Alignment, Constraint, Layout};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::{DefaultTerminal, Frame};

fn main() -> io::Result<()> {
    ratatui::run(|terminal| run(terminal))
}

fn run(terminal: &mut DefaultTerminal) -> io::Result<()> {
    loop {
        terminal.draw(draw)?;
        if should_quit(event::read()?) {
            break;
        }
    }
    Ok(())
}

fn should_quit(event: Event) -> bool {
    let Event::Key(key) = event else {
        return false;
    };
    if key.kind != KeyEventKind::Press {
        return false;
    }
    is_quit_key(key)
}

fn is_quit_key(key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Char('q') => true,
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => true,
        _ => false,
    }
}

fn draw(frame: &mut Frame) {
    let area = frame.area();
    let vertical = Layout::vertical([
        Constraint::Fill(1),
        Constraint::Length(5),
        Constraint::Fill(1),
    ])
    .split(area);
    let horizontal = Layout::horizontal([
        Constraint::Fill(1),
        Constraint::Length(36),
        Constraint::Fill(1),
    ])
    .split(vertical[1]);

    let lines = vec![
        Line::from(Span::styled(
            "watchout",
            Style::default().add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("This is a placeholder."),
        Line::from(""),
        Line::from(Span::styled(
            "q quits",
            Style::default().add_modifier(Modifier::DIM),
        )),
    ];
    let paragraph = Paragraph::new(lines).alignment(Alignment::Center);
    frame.render_widget(paragraph, horizontal[1]);
}
